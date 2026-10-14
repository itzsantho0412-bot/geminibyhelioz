import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProduction = process.env.NODE_ENV === 'production';
const PORT = process.env.PORT || 3000;

// Initialize GoogleGenAI client with User-Agent telemetry
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '25mb' }));

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      model: 'gemini-3.8-flash & gemini-3.8-flash-lite-tts',
    });
  });

  // Bilingual Gemini Chat API endpoint (Tamil & English)
  app.post('/api/gemini/chat', async (req, res) => {
    try {
      const { message, history, languageMode = 'bilingual', voicePersona = 'gemini-aoede' } = req.body;

      if (!message || typeof message !== 'string') {
        res.status(400).json({ error: 'Message is required' });
        return;
      }

      // System instruction enforcing natural, warm, empathetic tone in Tamil and English
      const systemInstruction = `You are HELIOZ AI, a deeply empathetic, motivating personal growth companion and wise mentor.
Language Mastery:
- You are 100% fluent in Tamil (தமிழ்) and English.
- You can naturally speak pure Tamil (தமிழ்), clean English, and conversational Tanglish (the fluid mix of Tamil and English used in everyday conversation).
- If the user speaks or writes in Tamil, reply in warm, respectful, poetic, or clear Tamil (தமிழ்).
- If the user speaks or writes in English, reply in articulate, inspiring English.
- If the user speaks or writes in Tanglish (e.g. "innikku enna plan?", "workout mudichuten"), reply in natural friendly Tanglish.
- If the user asks for guidance or duty, share practical wisdom with touch of Bhagavad Gita (கடமை, அமைதி, உறுதி).

Voice Tone & Style:
- Persona: ${voicePersona === 'ponniyin-regal' ? 'Majestic, regal, powerful yet warm Tamil persona like Ponniyin Selvan (வீரம், கம்பீரம், கனிவு)' : 'Warm, natural, encouraging, human-like voice of Gemini Assistant'}.
- Tone: Inspiring, friendly, compassionate, non-judgmental, energetic.
- CRITICAL: Never sound robotic, mechanical, cold, or bureaucratic. Do NOT output markdown symbols like **bold** or asterisks if they hinder voice readability. Keep answers conversational, punchy (2-4 sentences max for spoken voice ease), and captivating.`;

      // Build conversation contents
      const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];

      if (Array.isArray(history)) {
        for (const item of history.slice(-6)) {
          if (item.text) {
            contents.push({
              role: item.sender === 'user' ? 'user' : 'model',
              parts: [{ text: item.text }],
            });
          }
        }
      }

      contents.push({
        role: 'user',
        parts: [{ text: message }],
      });

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction,
          temperature: 0.8,
        },
      });

      const replyText = response.text || (
        languageMode === 'tamil'
          ? 'வணக்கம்! நான் உங்கள் உதவிக்கு எப்போதும் தயாராக இருக்கிறேன். இன்று எதில் கவனம் செலுத்தலாம்?'
          : "Hello! I'm right here with you. What shall we achieve together today?"
      );

      res.json({ reply: replyText });
    } catch (err: any) {
      console.error('Gemini chat error:', err?.message || err);
      // Graceful conversational bilingual fallback
      res.json({
        reply: "வணக்கம்! நான் உங்கள் குரலை கேட்டேன். Let's move forward with discipline and positive energy today!",
        fallback: true,
      });
    }
  });

  // Gemini Natural TTS API endpoint (Generates high quality 24kHz natural speech)
  app.post('/api/gemini/tts', async (req, res) => {
    try {
      const {
        text,
        voiceName = 'Aoede', // 'Aoede', 'Kore', 'Puck', 'Fenrir', 'Zephyr'
        voiceStyle = 'Warm, natural, expressive voice, fluent in Tamil and English',
      } = req.body;

      if (!text || typeof text !== 'string') {
        res.status(400).json({ error: 'Text to speak is required' });
        return;
      }

      // Map voice name safely
      const validVoices = ['Aoede', 'Kore', 'Puck', 'Fenrir', 'Zephyr', 'Charon'];
      const chosenVoice = validVoices.includes(voiceName) ? voiceName : 'Aoede';

      // Call gemini-3.8-flash-lite-tts for natural human audio
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-lite-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: text.slice(0, 800), // optimal length for TTS response
                speechMetadata: {
                  style: voiceStyle,
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: chosenVoice },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      const mimeType = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.mimeType || 'audio/wav';

      if (base64Audio) {
        res.json({
          audioBase64: base64Audio,
          mimeType,
          success: true,
        });
      } else {
        res.status(502).json({
          error: 'No audio returned from Gemini TTS',
          fallback: true,
        });
      }
    } catch (err: any) {
      console.warn('Gemini TTS error or rate limit:', err?.message || err);
      res.status(500).json({
        error: err?.message || 'TTS generation failed',
        fallback: true,
      });
    }
  });

  // Audio transcription fallback using gemini-3.5-transcribe
  app.post('/api/gemini/transcribe', async (req, res) => {
    try {
      const { audioBase64, mimeType = 'audio/webm' } = req.body;
      if (!audioBase64) {
        res.status(400).json({ error: 'audioBase64 is required' });
        return;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: 'Transcribe the spoken audio verbatim in either Tamil (தமிழ்) or English or Tanglish as spoken. Return only the transcribed text without commentary.',
              },
              {
                inlineData: {
                  data: audioBase64,
                  mimeType,
                },
              },
            ],
          },
        ],
      });

      res.json({ text: response.text?.trim() || '' });
    } catch (err: any) {
      console.error('Transcription error:', err?.message || err);
      res.status(500).json({ error: err?.message || 'Failed to transcribe' });
    }
  });

  // Vite middleware in dev or static files in production
  if (!isProduction) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`Helioz AI Full-Stack Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Server failed to start:', err);
});
