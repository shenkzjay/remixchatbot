import { useState, type SetStateAction, useEffect, type FormEvent } from "react";
import { Form, useActionData, useSubmit } from "react-router";
import type { ChatResponse } from "~/routes/home";

// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

interface MessageProps {
  role: "user" | "ai";
  content: string;
}

export default function ChatPopup() {
  const [toggle, setToggle] = useState(false);
  const [messages, setMessages] = useState<MessageProps[]>([]);

  const actionData = useActionData() as ChatResponse;
  const submit = useSubmit();

  useEffect(() => {
    if (actionData?.aiResponse) {
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: "ai", content: actionData.aiResponse as string },
      ]);
    }
  }, [actionData]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);

    const userMessage = formData.get("message") as string;

    if (userMessage.trim()) {
      setMessages((prevMessages) => [...prevMessages, { role: "user", content: userMessage }]);
      setMessages((prevMessages) => [...prevMessages, { role: "ai", content: "" }]);
      submit(formData, { method: "post" });
    }

    const response = await fetch("");

    // Reset the form
    e.currentTarget.reset();
  };

  return (
    <section className="absolute bottom-0 right-0 mx-6 mb-20 w-[25rem]">
      <div className="relative border h-full p-6">
        {toggle ? (
          <div className=" h-[30rem] flex flex-col justify-between">
            <div className="flex flex-row justify-between">
              <h3>Chatbot message</h3>
              <button onClick={() => setToggle(false)}>close</button>
            </div>
            <div className="h-[30rem] overflow-y-auto">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`${
                    message.role === "user"
                      ? "flex justify-end text-right"
                      : "flex justify-start text-left"
                  }`}
                >
                  <span
                    className={`flex  rounded-lg text-sm mb-6 ${
                      message.role === "user"
                        ? "bg-gray-200 text-gray-800 w-fit p-2"
                        : "bg-blue-100 text-blue-800 w-fit p-2"
                    }`}
                  >
                    {message.content}
                  </span>
                </div>
              ))}
            </div>
            <Form method="post" onSubmit={(e) => handleSubmit(e)}>
              <label className="sr-only">Message</label>
              <input type="text" placeholder="Message..." id="message" name="message" />
              <button type="submit">Send</button>
            </Form>
          </div>
        ) : (
          <button onClick={() => setToggle(!toggle)} className="anchor">
            Pipup button
          </button>
        )}
      </div>
    </section>
  );
}
