import OpenAI from 'openai';
import { supabase } from './supabase';

const openai = new OpenAI({
  apiKey: import.meta.env.VITE_OPENAI_API_KEY,
  dangerouslyAllowBrowser: true // Only for demo - use edge functions in production
});

export interface TrainingData {
  text: string[];
  audio: string[];
  video: string[];
  images: string[];
}

export interface PersonaProfile {
  personality: string;
  speechPatterns: string[];
  commonPhrases: string[];
  emotionalTone: string;
  memories: string[];
  voiceCharacteristics?: {
    pitch: number;
    speed: number;
    tone: string;
  };
}

export class AITrainingPipeline {
  private personaId: string;
  private onProgress: (step: string, progress: number) => void;

  constructor(personaId: string, onProgress: (step: string, progress: number) => void) {
    this.personaId = personaId;
    this.onProgress = onProgress;
  }

  async startTraining(): Promise<PersonaProfile> {
    try {
      // Step 1: Gather training data
      this.onProgress('content-analysis', 10);
      const trainingData = await this.gatherTrainingData();

      // Step 2: Process text content
      this.onProgress('content-analysis', 30);
      const textAnalysis = await this.analyzeTextContent(trainingData.text);

      // Step 3: Process audio content
      this.onProgress('voice-modeling', 50);
      const voiceAnalysis = await this.analyzeAudioContent(trainingData.audio);

      // Step 4: Process visual content
      this.onProgress('personality-extraction', 70);
      const visualAnalysis = await this.analyzeVisualContent(trainingData.images, trainingData.video);

      // Step 5: Create unified persona profile
      this.onProgress('conversation-training', 85);
      const personaProfile = await this.createPersonaProfile(textAnalysis, voiceAnalysis, visualAnalysis);

      // Step 6: Fine-tune conversation model
      this.onProgress('final-optimization', 95);
      await this.optimizeConversationModel(personaProfile);

      this.onProgress('final-optimization', 100);
      return personaProfile;

    } catch (error) {
      console.error('Training pipeline error:', error);
      throw new Error('Failed to train AI persona');
    }
  }

  private async gatherTrainingData(): Promise<TrainingData> {
    const { data: content, error } = await supabase
      .from('persona_content')
      .select('*')
      .eq('persona_id', this.personaId)
      .eq('processing_status', 'completed');

    if (error) throw error;

    const trainingData: TrainingData = {
      text: [],
      audio: [],
      video: [],
      images: []
    };

    for (const item of content || []) {
      switch (item.content_type) {
        case 'text':
          if (item.content_text) {
            trainingData.text.push(item.content_text);
          }
          break;
        case 'audio':
          if (item.file_url) {
            trainingData.audio.push(item.file_url);
          }
          break;
        case 'video':
          if (item.file_url) {
            trainingData.video.push(item.file_url);
          }
          break;
        case 'image':
          if (item.file_url) {
            trainingData.images.push(item.file_url);
          }
          break;
      }
    }

    return trainingData;
  }

  private async analyzeTextContent(textData: string[]): Promise<any> {
    if (textData.length === 0) return null;

    const combinedText = textData.join('\n\n');
    
    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: `Analyze this person's text content and extract:
          1. Personality traits and characteristics
          2. Common phrases and expressions they use
          3. Communication style and tone
          4. Emotional patterns
          5. Important memories and experiences mentioned
          6. Values and beliefs expressed
          
          Return a detailed JSON analysis with these categories.`
        },
        { role: 'user', content: combinedText }
      ],
      max_tokens: 1500,
      temperature: 0.3
    });

    try {
      return JSON.parse(completion.choices[0]?.message?.content || '{}');
    } catch {
      return { error: 'Failed to parse text analysis' };
    }
  }

  private async analyzeAudioContent(audioUrls: string[]): Promise<any> {
    if (audioUrls.length === 0) return null;

    // For each audio file, transcribe and analyze
    const transcriptions = [];
    
    for (const audioUrl of audioUrls.slice(0, 5)) { // Limit to first 5 files
      try {
        // Download audio file
        const response = await fetch(audioUrl);
        const audioBlob = await response.blob();
        
        // Convert to File object for OpenAI
        const audioFile = new File([audioBlob], 'audio.mp3', { type: 'audio/mpeg' });
        
        // Transcribe audio
        const transcription = await openai.audio.transcriptions.create({
          file: audioFile,
          model: 'whisper-1',
          response_format: 'text'
        });
        
        transcriptions.push(transcription);
      } catch (error) {
        console.error('Audio transcription error:', error);
      }
    }

    if (transcriptions.length === 0) return null;

    // Analyze transcribed speech patterns
    const combinedTranscription = transcriptions.join('\n\n');
    
    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: `Analyze this person's speech patterns from transcribed audio:
          1. Speaking style and rhythm
          2. Vocabulary preferences
          3. Emotional expression in speech
          4. Common verbal habits or filler words
          5. Tone and mood patterns
          6. Voice characteristics that can be described
          
          Return a JSON analysis focusing on speech and voice patterns.`
        },
        { role: 'user', content: combinedTranscription }
      ],
      max_tokens: 1000,
      temperature: 0.3
    });

    try {
      return JSON.parse(completion.choices[0]?.message?.content || '{}');
    } catch {
      return { error: 'Failed to parse audio analysis' };
    }
  }

  private async analyzeVisualContent(imageUrls: string[], videoUrls: string[]): Promise<any> {
    const visualAnalysis = {
      expressions: [],
      settings: [],
      activities: [],
      relationships: []
    };

    // Analyze images using GPT-4 Vision
    for (const imageUrl of imageUrls.slice(0, 10)) { // Limit to first 10 images
      try {
        const completion = await openai.chat.completions.create({
          model: 'gpt-4-vision-preview',
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text: 'Analyze this image and describe: 1) The person\'s facial expression and body language, 2) The setting/environment, 3) Any activities or interactions visible, 4) The overall mood or emotion conveyed. Be specific but respectful.'
                },
                {
                  type: 'image_url',
                  image_url: { url: imageUrl }
                }
              ]
            }
          ],
          max_tokens: 300
        });

        const analysis = completion.choices[0]?.message?.content;
        if (analysis) {
          visualAnalysis.expressions.push(analysis);
        }
      } catch (error) {
        console.error('Image analysis error:', error);
      }
    }

    return visualAnalysis;
  }

  private async createPersonaProfile(textAnalysis: any, voiceAnalysis: any, visualAnalysis: any): Promise<PersonaProfile> {
    // Combine all analyses into a unified persona profile
    const combinedData = {
      textAnalysis: textAnalysis || {},
      voiceAnalysis: voiceAnalysis || {},
      visualAnalysis: visualAnalysis || {}
    };

    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: `Create a comprehensive AI persona profile from the provided analysis data. 
          Generate a unified personality description, list of common phrases, speech patterns, 
          emotional characteristics, and key memories. Format as JSON with these fields:
          - personality: string (detailed personality description)
          - speechPatterns: string[] (how they speak and express themselves)
          - commonPhrases: string[] (specific phrases they commonly use)
          - emotionalTone: string (overall emotional characteristics)
          - memories: string[] (important memories and experiences)
          - voiceCharacteristics: object with pitch, speed, tone`
        },
        { role: 'user', content: JSON.stringify(combinedData) }
      ],
      max_tokens: 1500,
      temperature: 0.4
    });

    try {
      const profile = JSON.parse(completion.choices[0]?.message?.content || '{}');
      
      // Save profile to database
      await supabase
        .from('personas')
        .update({
          personality_traits: profile.personality,
          common_phrases: profile.commonPhrases || [],
          status: 'active',
          training_progress: 100,
          updated_at: new Date().toISOString()
        })
        .eq('id', this.personaId);

      return profile;
    } catch (error) {
      console.error('Profile creation error:', error);
      throw new Error('Failed to create persona profile');
    }
  }

  private async optimizeConversationModel(profile: PersonaProfile): Promise<void> {
    // Create a system prompt for this specific persona
    const systemPrompt = `You are embodying a specific person with these characteristics:

PERSONALITY: ${profile.personality}

SPEECH PATTERNS: ${profile.speechPatterns?.join(', ')}

COMMON PHRASES: Use these naturally in conversation: ${profile.commonPhrases?.join(', ')}

EMOTIONAL TONE: ${profile.emotionalTone}

MEMORIES: Reference these experiences when relevant: ${profile.memories?.join('; ')}

IMPORTANT: Always respond as this specific person, using their unique voice, mannerisms, and personality. Be authentic to their character while being helpful and emotionally supportive.`;

    // Store the optimized prompt for use in conversations
    await supabase
      .from('personas')
      .update({
        metadata: {
          systemPrompt,
          trainingCompleted: new Date().toISOString(),
          voiceCharacteristics: profile.voiceCharacteristics
        }
      })
      .eq('id', this.personaId);
  }
}

// Helper function to start training for a persona
export async function trainPersonaFromUploadedContent(
  personaId: string,
  onProgress: (step: string, progress: number) => void
): Promise<PersonaProfile> {
  const pipeline = new AITrainingPipeline(personaId, onProgress);
  return await pipeline.startTraining();
}