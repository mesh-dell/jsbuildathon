import { AzureOpenAI } from "openai";
import dotenv from "dotenv";

dotenv.config();

export async function main() {
  // You will need to set these environment variables or edit the following values
  const endpoint = process.env["AZURE_OPENAI_ENDPOINT"];
  const apiKey = process.env["AZURE_OPENAI_API_KEY"];
  const apiVersion = "2025-01-01-preview";
  const deployment = "o3-mini"; // This must match your deployment name

  const client = new AzureOpenAI({ endpoint, apiKey, apiVersion, deployment });

  const result = await client.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are an AI assistant that helps people find information.",
      },
      { role: "user", content: "I am going to Paris, what should I see?" },
      {
        role: "assistant",
        content:
          "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.",
      },
      { role: "user", content: "What is so great about #1?" },
    ],
    max_completion_tokens: 800,
  });

  console.log(JSON.stringify(result, null, 2));
  console.log("Response:", result.choices[0].message.content);
}

main().catch((err) => {
  console.error("The sample encountered an error:", err);
});
