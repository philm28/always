import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn('Missing Supabase environment variables. Using placeholder values for development. Please configure your .env file with actual Supabase credentials.')
}

export const supabase = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseAnonKey || 'placeholder-anon-key'
)

// Database types
export interface Profile {
  id: string
  user_id: string
  email: string
  full_name?: string
  avatar_url?: string
  subscription_tier: string
  created_at: string
  updated_at: string
}

export interface Persona {
  id: string
  user_id: string
  name: string
  relationship?: string
  description?: string
  personality_traits?: string
  common_phrases?: string[]
  avatar_url?: string
  voice_model_id?: string
  status: 'training' | 'active' | 'error'
  training_progress: number
  created_at: string
  updated_at: string
}

export interface PersonaContent {
  id: string
  persona_id: string
  content_type: 'video' | 'audio' | 'text' | 'image' | 'social_media'
  file_url?: string
  file_name?: string
  file_size?: number
  content_text?: string
  metadata: Record<string, any>
  processing_status: 'pending' | 'processing' | 'completed' | 'error'
  created_at: string
}

export interface Conversation {
  id: string
  user_id: string
  persona_id: string
  title?: string
  conversation_type: 'chat' | 'video_call' | 'voice_call'
  duration_seconds: number
  started_at: string
  ended_at?: string
  metadata: Record<string, any>
}

export interface Message {
  id: string
  conversation_id: string
  sender_type: 'user' | 'persona'
  content: string
  message_type: 'text' | 'audio' | 'video'
  audio_url?: string
  video_url?: string
  timestamp: string
  metadata: Record<string, any>
}