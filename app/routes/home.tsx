import ChatPopup from "~/chatpop-up/chat";
import type { Route } from "./+types/home";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { type MessageContent, HumanMessage, SystemMessage } from "@langchain/core/messages";

export interface ChatResponse {
  aiResponse: MessageContent;
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}
export const action = async ({ request }: Route.ActionArgs) => {
  const formData = await request.formData();

  const usermessage = formData.get("message") as string;

  console.log(usermessage);

  const message = usermessage.trim();

  if (!message) {
    return {
      aiResponse: "**message field empty**",
    };
  }

  console.log(message);

  const gemini = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-1.5-flash",
    maxOutputTokens: 100,
    temperature: 0.8,
    maxRetries: 2,
  });

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", "You an assitant, that answers questions"],
    new MessagesPlaceholder("history"),
    ["human", "{message}"],
  ]);

  const memory = new BufferMemory({
    returnMessages: true,
    memoryKey: "history",
  });

  const chain = new ConversationChain({
    llm: gemini,
    memory: memory,
    prompt: promptTemplate,
    outputParser: new StringOutputParser(),
  });

  // const stream = await chain.stream({
  //   message: message,
  // });

  const result = await chain.invoke({
    message: message,
  });

  // console.log("stream", stream);
  // console.log("result", result);

  return {
    aiResponse: result.response,
  };
};

export default function Home() {
  return (
    <section className="bg-slate-100 h-screen w-full relative">
      <div>
        <ChatPopup />
      </div>
    </section>
  );
}
