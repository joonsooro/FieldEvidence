import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function generateCode() {
  const response = await client.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      { role: "system", content: "You are an expert coding assistant." },
      { role: "user", content: "Write a Python function that reverses a string." }
    ],
  });

  console.log(response.choices[0].message.content);
}

generateCode();
