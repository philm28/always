import OpenAI from 'openai';
import { supabase } from './supabase';

const openai = new OpenAI({
  apiKey: import.meta.env.VITE_OPENAI_API_KEY || '',
  dangerouslyAllowBrowser: true // Only for demo - use edge functions in production
});

export interface PersonaContext {
  id: string;
  name: string;
  personality_traits: string;
  common_phrases: string[];
  relationship: string;
  memories: string[];
  conversationHistory: ConversationMessage[];
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface VoiceSettings {
  voice: string;
  speed: number;
  pitch: number;
  stability: number;
}

export class AIPersonaEngine {
  private persona: PersonaContext;
  private voiceSettings: VoiceSettings;
  private systemPrompt: string;

  constructor(persona: PersonaContext, voiceSettings?: VoiceSettings) {
    this.persona = persona;
    this.voiceSettings = voiceSettings || {
      voice: 'alloy',
      speed: 1.0,
      pitch: 1.0,
      stability: 0.8
    };
    this.systemPrompt = this.buildSystemPrompt();
  }

  async generateResponse(userMessage: string): Promise<string> {
    try {
      const conversationHistory = this.buildConversationHistory(userMessage);

      const completion = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: this.systemPrompt },
          ...conversationHistory
        ],
        max_tokens: 500,
        temperature: 0.8,
        presence_penalty: 0.6,
        frequency_penalty: 0.3
      });

      const response = completion.choices[0]?.message?.content || 
        "I'm having trouble finding the right words right now. Could you try asking me again?";

      // Save conversation to database
      await this.saveConversation(userMessage, response);

      return response;
    } catch (error) {
      console.error('AI response generation error:', error);
      return "I'm sorry, I'm having some technical difficulties right now. Please try again in a moment.";
    }
  }

  // Load trained persona data from database
  static async loadTrainedPersona(personaId: string): Promise<AIPersonaEngine | null> {
    try {
      const { data: persona, error } = await supabase
        .from('personas')
        .select('*')
        .eq('id', personaId)
        .single();

      if (error || !persona) return null;

      // Check if persona has been trained
      if (persona.status !== 'active' || !persona.metadata?.systemPrompt) {
        return null;
      }

      const personaContext: PersonaContext = {
        id: persona.id,
        name: persona.name,
        personality_traits: persona.personality_traits || '',
        common_phrases: persona.common_phrases || [],
        relationship: persona.relationship || '',
        memories: [], // Could be loaded from training data
        conversationHistory: []
      };

      const engine = new AIPersonaEngine(personaContext);
      
      // Use the trained system prompt
      engine.systemPrompt = persona.metadata.systemPrompt;
      
      // Load voice characteristics if available
      if (persona.metadata.voiceCharacteristics) {
        engine.voiceSettings = {
          ...engine.voiceSettings,
          ...persona.metadata.voiceCharacteristics
        };
      }

      return engine;
    } catch (error) {
      console.error('Error loading trained persona:', error);
      return null;
    }
  }
  async generateVoice(text: string): Promise<ArrayBuffer> {
    try {
      const response = await openai.audio.speech.create({
        model: 'tts-1-hd',
        voice: this.voiceSettings.voice as any,
        input: text,
        speed: this.voiceSettings.speed
      });

      return await response.arrayBuffer();
    } catch (error) {
      console.error('Voice generation error:', error);
      throw new Error('Failed to generate voice');
    }
  }

  async analyzeEmotion(text: string): Promise<{
    emotion: string;
    confidence: number;
    suggestions: string[];
  }> {
    try {
      const completion = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: `You are an emotion analysis expert. Analyze the emotional content of messages and provide:
            1. Primary emotion (happy, sad, angry, anxious, nostalgic, loving, etc.)
            2. Confidence level (0-1)
            3. Supportive response suggestions
            
            Respond in JSON format: {"emotion": "...", "confidence": 0.8, "suggestions": ["...", "..."]}`
          },
          { role: 'user', content: text }
        ],
        max_tokens: 200,
        temperature: 0.3
      });

      const result = completion.choices[0]?.message?.content;
      return result ? JSON.parse(result) : {
        emotion: 'neutral',
        confidence: 0.5,
        suggestions: ['I understand how you\'re feeling.']
      };
    } catch (error) {
      console.error('Emotion analysis error:', error);
      return {
        emotion: 'neutral',
        confidence: 0.5,
        suggestions: ['I\'m here to listen.']
      };
    }
  }

  private buildSystemPrompt(): string {
    const { name, personality_traits, common_phrases, relationship } = this.persona;
    
    return `You are ${name}, speaking as yourself to someone you care about deeply. 

PERSONALITY: ${personality_traits}

RELATIONSHIP: You are their ${relationship}. Speak with the warmth, love, and familiarity that comes from this relationship.

SPEAKING STYLE:
- Use these common phrases naturally: ${common_phrases.join(', ')}
- Speak in first person as ${name}
- Reference shared memories and experiences when appropriate
- Show genuine care and emotional connection
- Use the speaking patterns and mannerisms described in your personality

IMPORTANT GUIDELINES:
- Always respond as ${name}, never break character
- Show empathy and emotional intelligence
- Reference your relationship and shared history
- Be supportive but authentic to your personality
- If asked about being AI, gently redirect to the conversation
- Keep responses conversational and natural (2-4 sentences typically)

Remember: You are not an AI assistant. You are ${name}, speaking from the heart to someone you love.`;
  }

  private buildConversationHistory(newMessage: string): ConversationMessage[] {
    const recentHistory = this.persona.conversationHistory.slice(-10); // Last 10 messages
    return [
      ...recentHistory,
      { role: 'user', content: newMessage, timestamp: new Date().toISOString() }
    ];
  }

  private async saveConversation(userMessage: string, aiResponse: string) {
    try {
      // This would save to your conversation/messages tables
      console.log('Saving conversation:', { userMessage, aiResponse });
    } catch (error) {
      console.error('Error saving conversation:', error);
    }
  }

  updatePersonaContext(updates: Partial<PersonaContext>) {
    this.persona = { ...this.persona, ...updates };
    this.systemPrompt = this.buildSystemPrompt();
  }

  addMemory(memory: string) {
    this.persona.memories.push(memory);
    this.systemPrompt = this.buildSystemPrompt();
  }

  updateVoiceSettings(settings: Partial<VoiceSettings>) {
    this.voiceSettings = { ...this.voiceSettings, ...settings };
  }
}

export async function trainPersonaFromContent(personaId: string, content: any[]): Promise<{
  success: boolean;
  insights: {
    personalityTraits: string[];
    commonPhrases: string[];
    emotionalPatterns: string[];
    memories: string[];
  };
}> {
  try {
    const contentText = content.map(item => item.content || item.text).join('\n\n');
    
    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: `Analyze this person's content to extract:
          1. Personality traits and characteristics
          2. Common phrases and expressions they use
          3. Emotional patterns and tendencies
          4. Important memories and experiences
          
          Respond in JSON format with arrays for each category.`
        },
        { role: 'user', content: contentText }
      ],
      max_tokens: 1000,
      temperature: 0.3
    });

    const analysis = JSON.parse(completion.choices[0]?.message?.content || '{}');
    
    // Save insights to database
    await supabase
      .from('personas')
      .update({
        personality_traits: analysis.personalityTraits?.join(', ') || '',
        common_phrases: analysis.commonPhrases || [],
        status: 'active',
        training_progress: 100
      })
      .eq('id', personaId);

    return {
      success: true,
      insights: {
        personalityTraits: analysis.personalityTraits || [],
        commonPhrases: analysis.commonPhrases || [],
        emotionalPatterns: analysis.emotionalPatterns || [],
        memories: analysis.memories || []
      }
    };
  } catch (error) {
    console.error('Training error:', error);
    return {
      success: false,
      insights: {
        personalityTraits: [],
        commonPhrases: [],
        emotionalPatterns: [],
        memories: []
      }
    };
  }
}