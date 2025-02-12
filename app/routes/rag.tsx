import { Form, useActionData, useSubmit } from "react-router";
import type { Route } from "../+types/root";
import { ChatMistralAI } from "@langchain/mistralai";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { concat } from "@langchain/core/utils/stream";
import { StateGraph } from "@langchain/langgraph";

import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "node_modules/@langchain/textsplitters/dist";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { PineconeTranslator } from "@langchain/pinecone";
import { useEffect, useState, useRef } from "react";
import { AddIcon } from "public/icons/add";
import Markdown from "react-markdown";

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();

  const file = formData.get("uploadfile") as File;

  const query = formData.get("_query");

  const intent = formData.get("intent");

  const questions = formData.get("questions") as string;

  // console.log({ question });

  // let vectorStore: MemoryVectorStore | null = null;

  const formatDocumentsAsString = (documents: Document[]) => {
    return documents.map((document) => document.pageContent).join("\n\n");
  };

  const llm = new ChatMistralAI({
    apiKey: process.env.MISTRAL_API_KEY,
    model: "mistral-large-latest",
    temperature: 0.8,
    maxRetries: 2,
  });

  const indexName = "quickstart";

  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!,
  });

  const pineconeIndex = pinecone.Index(indexName);

  if (query === "upload") {
    //pdf document loader
    const fileArrayBuffer = await file.arrayBuffer();

    const fileLoader = new PDFLoader(new Blob([fileArrayBuffer]));

    const pdf = await fileLoader.load();

    // console.log("pdf", pdf[0]);

    //split the text in the document into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await splitter.splitDocuments(pdf);

    chunks.forEach((chunk) => {
      chunk.metadata = {
        ...chunk.metadata,
        filename: file.name,
      };
    });

    console.log(chunks);

    const embeddings = new MistralAIEmbeddings({
      model: "mistral-embed",
    });

    const vectorStore = await PineconeStore.fromDocuments(chunks, embeddings, {
      pineconeIndex: pineconeIndex,
    });

    console.log(vectorStore);

    //embedded the chunks and store the datapoints in a memorystore

    // await vectorStore.addDocuments(chunks);

    // return vectorStore;
  }

  if (query === "questions") {
    const indexName = "quickstart";

    const pineconeIndex = pinecone.Index(indexName);

    const embeddings = new MistralAIEmbeddings({
      model: "mistral-embed",
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: pineconeIndex,
    });

    const customPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You provide helpful assistant. Use the context below to answer questions. Follow these rules:
       
        1. Keep answers under 3 paragraphs
        
        Context: {context}`,
      ],
      ["human", "Question: {question}"],
    ]);

    const promptTemplate = customPrompt;

    await promptTemplate.invoke({
      context: "(context goes here)",
      question: "(question goes here)",
    });

    const InputStateAnnotation = Annotation.Root({
      question: Annotation<string>,
    });

    const StateAnnotation = Annotation.Root({
      question: Annotation<string>,
      context: Annotation<Document[]>,
      answer: Annotation<string>,
    });

    const retrieve = async (state: typeof InputStateAnnotation.State) => {
      const retrievedDocs = await vectorStore?.similaritySearch(state.question, 3);

      console.log("Retrieved documents:", retrievedDocs);
      return { context: retrievedDocs };
    };

    const generate = async (state: typeof StateAnnotation.State) => {
      const docsContent = state?.context?.map((doc) => doc.pageContent).join("\n");
      const messages = await promptTemplate.invoke({
        question: state.question,
        context: docsContent,
      });

      const response = await llm.invoke(messages);

      return { answer: response.content };
    };

    const graph = new StateGraph(StateAnnotation)
      .addNode("retrieve", retrieve)
      .addNode("generate", generate)
      .addEdge("__start__", "retrieve")
      .addEdge("retrieve", "generate")
      .addEdge("generate", "__end__")
      .compile();

    let inputs = { question: questions };

    const result = await graph.invoke(inputs);
    console.log(result?.context?.slice(0, 2));
    console.log(`\nAnswer: ${result["answer"]}`);

    return { result };
  }

  if (query === "deletefile") {
    console.log(file?.name);

    const filebtn = formData.get("btnfile") as string;

    console.log(filebtn);

    try {
      const deleteFile = await pineconeIndex.deleteMany({
        filename: { $eq: filebtn },
      });

      console.log({ deleteFile });
    } catch (error) {
      console.log(error);
    }
  }
}

export default function Rag() {
  interface MessageProps {
    role: "user" | "ai";
    content: string;
  }

  const [file, setFileUpload] = useState<File | null>(null);
  const [allFiles, setAllfileUpload] = useState<string[]>([]);
  const [toggleDisplay, setToggleDisplay] = useState(true);
  const uploadFormRef = useRef<HTMLFormElement | null>(null);
  const submitUploadRef = useRef<HTMLFormElement | null>(null);
  const [textareaInput, setTextareaInput] = useState("");
  const [message, setMessage] = useState<MessageProps[]>([]);
  const [toggleNav, setToggleNav] = useState(false);
  const [isloading, setIsLoading] = useState(false);
  const messageBoxRef = useRef<HTMLDivElement | null>(null);
  const touchStartX = useRef(0);
  const touchEndX = useRef(0);

  console.log({ message });

  const submit = useSubmit();

  const actionData = useActionData();

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (messageBoxRef.current) {
        messageBoxRef.current.scrollTo({
          top: messageBoxRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 0);

    return () => clearTimeout(timeout);
  }, [message, isloading]);

  useEffect(() => {
    if (actionData?.result) {
      setMessage((prevMessages) => [
        ...prevMessages,
        // { role: "user", content: actionData.result.question as string },
        { role: "ai", content: actionData.result.answer as string },
      ]);

      setIsLoading(false);
    } else {
      setIsLoading(false);
      return;
    }
  }, [actionData]);

  useEffect(() => {
    if (localStorage.length > 0) {
      const uploadedfile = localStorage.getItem("file");

      if (uploadedfile) {
        const parsedUploadfile = JSON.parse(uploadedfile);

        setAllfileUpload((prev) => [...prev, parsedUploadfile]);
      }
    }
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFileUpload(e.target.files[0]);

      // setAllfileUpload((prev) => [...prev, e.target.files![0]]);
    }
  };

  const handleDeleteFile = (file: string, e: React.MouseEvent<HTMLButtonElement>) => {
    // setAllfileUpload(allFiles.filter((files, _) => files !== file));
    // localStorage.removeItem("file");
    setToggleDisplay(false);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const loadingTimeout = setTimeout(() => setIsLoading(true), 300);

    if (uploadFormRef.current) {
      const formData = new FormData(e.currentTarget);
      setToggleNav(false);
      const userQuestion = formData.get("questions") as string;
      formData.append("_query", "questions");

      if (userQuestion.trim()) {
        setMessage((prev) => [...prev, { role: "user", content: userQuestion }]);
        // setMessage((prev) => [...prev, { role: "ai", content: "" }]);
        submit(formData, { method: "post" });
      } else {
        setIsLoading(false);
        return;
      }

      // setToggleNav(true);
    }

    e.currentTarget?.reset();

    return () => {
      clearTimeout(loadingTimeout);
    };
  };

  const handleSubmitFile = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (submitUploadRef.current) {
      const formData = new FormData(submitUploadRef.current);

      const uploadedfiles = formData.get("uploadfile") as File | null;

      if (!uploadedfiles) return console.log("please upload file");

      formData.append("uploadfile", uploadedfiles);
      formData.append("_query", "upload");

      console.log("Submitting file:", uploadedfiles);

      setAllfileUpload((prev) => [...prev, uploadedfiles?.name]);

      submit(formData, { method: "post", encType: "multipart/form-data" });

      const existingFiles: string[] = JSON.parse(localStorage.getItem("file") || "[]");

      const updatedFiles = Array.isArray(existingFiles)
        ? [...existingFiles, uploadedfiles.name]
        : [uploadedfiles.name];

      localStorage.setItem("file", JSON.stringify(updatedFiles));
    }
  };

  const handleToggleAddButton = () => {
    setToggleNav(!toggleNav);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    touchEndX.current = e.touches[0].clientX;
  };

  const handleTouchEnd = () => {
    const diff = touchEndX.current - touchStartX.current;

    if (diff > 50) {
      setToggleNav(true);
    } else if (diff < -50) {
      setToggleNav(false);
    }
  };

  console.log({ allFiles });
  console.log({ message });

  return (
    <section className="flex bg-[#262626] relative">
      {/* side menubar */}
      <div className="md:hidden flex flex-row" role="button" onClick={handleToggleAddButton}>
        <button
          className={` ${
            toggleNav ? "[transform:rotate(45deg)]" : "[transform:rotate(0)]"
          } z-50 m-6 text-white absolute top-0 [transform-origin:0_0 ] [transition:_transform_300ms_linear] burst`}
        >
          <AddIcon />
        </button>
        <p className="absolute top-0 z-10 text-white m-6 pl-8">{toggleNav ? "" : "Upload doc "}</p>
      </div>
      <aside
        className="relative"
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <div
          className={` ${
            toggleNav
              ? "[transform:_translatex(0px)] w-[400px] h-full bg-[#333]"
              : "[transform:_translatex(-100vw)]  h-full bg-[#333] "
          } [transition:_transform_300ms_linear] md:[transform:_translatex(0px)] fixed md:static md:flex md:flex-col md:w-[15rem] w-full border-r p-6 border-r-[#333] pt-20 z-10`}
        >
          <Form
            method="post"
            className=""
            encType="multipart/form-data"
            onSubmit={handleSubmitFile}
            ref={submitUploadRef}
          >
            <div className="upload-wrapper success  w-full relative">
              <label className="sr-only">Upload file</label>
              <input type="hidden" name="_query" id="_query" value="upload" />
              <input
                required
                onChange={(e) => handleFileUpload(e)}
                type="file"
                name="uploadfile"
                id="uploadfile"
                className=" absolute top-0 left-0 right-0 bottom-0 opacity-0 cursor-pointer"
              />
              <div className="uploadzone success w-full p-[20px] border border-dashed border-[#666] text-center bg-[#333] [transition:_background-color_0.3s_ease-in-out] rounded-[10px] text-slate-400">
                <div className="default flex flex-col justify-center items-center gap-4">
                  <figure className="w-10 h-10 bg-[#f4f4f4] flex flex-row items-center justify-center rounded-full"></figure>
                  <pre className="text-[12px] text-wrap text-buttongray">
                    <b className="text-primary flex text-center justify-center">Click to upload</b>
                    or <br />
                    drag and drop Pdf (max. 25MB)
                  </pre>

                  {/* <button
                  type="button"
                  className="px-8 py-2 font-semibold rounded-[10px] border border-[#666] border-primary text-primary"
                >
                  Browse Files
                </button> */}
                </div>
                <span className="success text-buttongray flex py-12 min-h-full text-xs text-wrap w-full overflow-clip">
                  {file?.name} - {`${(file?.size! / 1024).toLocaleString()}kb`}
                  <br />
                  {/* <i className="text-primary">{}</i> */}
                </span>
              </div>
            </div>
            <div className="flex justify-center group">
              <button
                type="submit"
                className="group-hover:text-slate-200 mt-6 px-4 py-2 text-slate-400 rounded-xl bg-gray-700 text-sm"
              >
                Upload file
              </button>
            </div>
          </Form>

          {allFiles.length > 0 && (
            <div className="mt-24 flex flex-col gap-4">
              <span className="text-slate-400">List of uploaded files</span>
              <ul className=" flex flex-col gap-4  h-[18rem] overflow-y-scroll ">
                {allFiles &&
                  allFiles.map((file, idx) => (
                    <li
                      key={idx}
                      className="text-white flex flex-row gap-4 bg-[#333] py-1 px-1 rounded-xl w-fit text-xs justify-center items-center border border-slate-600"
                    >
                      <p> {`${file}` || null}</p>
                      <Form method="post">
                        <input type="hidden" id="btnfile" name="btnfile" value={file} />
                        <button
                          name="_query"
                          value="deletefile"
                          onClick={(e) => handleDeleteFile(file, e)}
                          className="bg-white/30 hover:bg-red-500 p-1 w-5 leading-none rounded-xl"
                        >
                          x
                        </button>
                      </Form>
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </div>
      </aside>
      <div className="h-[80vh] w-full flex flex-col justify-between items-center mb-32">
        <div
          ref={messageBoxRef}
          className={`text-white p-4 rounded-xl  overflow-y-scroll mt-12 md:w-[45vw] w-full `}
        >
          {message &&
            message.map((msg, idx) => (
              <div
                key={idx}
                className={`mb-6 ${
                  msg.role === "user" ? "flex justify-end text-left" : "text-left"
                }`}
              >
                <div
                  className={`inline-block max-w-[80%] rounded-lg p-2 ${
                    msg.role === "user" ? "  text-slate-300" : "bg-gray-300 text-gray-800"
                  } bg-[#333] p-2`}
                >
                  {msg.role === "user" ? (
                    <div className=" text-slate-300 text-sm text-left">{msg.content}</div>
                  ) : (
                    <Markdown className="text-sm prose ">{msg.content}</Markdown>
                  )}
                </div>
              </div>
            ))}

          {isloading && (
            <div className="bg-[#444] w-fit h-auto p-2 rounded-xl">
              <span className="loader flex"></span>
            </div>
          )}
        </div>

        <Form
          method="post"
          ref={uploadFormRef}
          onSubmit={handleSubmit}
          className=" w-full md:w-auto fixed md:static bottom-0 bg-[#262626]"
        >
          {message.length === 0 && (
            <p className="mt-6 mx-6 md:mx-0 text-slate-400 text-2xl">
              Who do you want to know about the doc?
            </p>
          )}
          <div className="flex flex-row my-6 mx-6 md:mx-0  justify-between  md:w-[40rem] items-end bg-[#333] p-4 rounded-xl">
            <input type="hidden" name="_query" id="_query" value="questions" />

            <textarea
              className="py-2  focus:border-0 focus:outline-0 min-h-12 resize-none ring-0 w-full bg-[#333]  border-[#666] px-2 rounded-xl text-slate-300"
              placeholder="Ask me"
              name="questions"
              id="questions"
              required
            ></textarea>
            <div className="flex flex-row justify-between mt-4">
              <button className=" rounded-xl text-white" type="submit">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 32 32"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  className="bg-black rounded-full"
                >
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M15.1918 8.90615C15.6381 8.45983 16.3618 8.45983 16.8081 8.90615L21.9509 14.049C22.3972 14.4953 22.3972 15.2189 21.9509 15.6652C21.5046 16.1116 20.781 16.1116 20.3347 15.6652L17.1428 12.4734V22.2857C17.1428 22.9169 16.6311 23.4286 15.9999 23.4286C15.3688 23.4286 14.8571 22.9169 14.8571 22.2857V12.4734L11.6652 15.6652C11.2189 16.1116 10.4953 16.1116 10.049 15.6652C9.60265 15.2189 9.60265 14.4953 10.049 14.049L15.1918 8.90615Z"
                    fill="currentColor"
                  ></path>
                </svg>
              </button>
            </div>
          </div>
        </Form>

        {/* <Form method="post" onSubmit={handleSubmit}></Form> */}
      </div>
    </section>
  );
}
