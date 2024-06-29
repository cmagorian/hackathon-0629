"use client";

import { Message, UserData } from "@/app/data";
import ChatTopbar from "./chat-topbar";
import { ChatList } from "./chat-list";
import React, { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import getMovies from "@/api/getMovies";

interface ChatProps {
  messages?: Message[];
  selectedUser: UserData;
  isMobile: boolean;
}

export function Chat({ messages, selectedUser, isMobile }: ChatProps) {
  function processNewlines(inputString: string) {
    /**
     * This function replaces occurrences of the escape sequence "\\n" in the input string with a literal newline character.
     * 
     * @param {string} inputString - The string to process.
     * @return {string} - The processed string with literal newlines.
     */
    return inputString.replace(/\\n/g, '\n');
}

  const [input, setInput] = useState<string>('1');

  const { data, isLoading, isError, refetch } = useQuery({
    queryFn: async () => await getMovies(input),
    queryKey: ["movies"], // Array according to Documentation
    enabled: false, // Disable automatic fetching
  });

  const [messagesState, setMessages] = React.useState<Message[]>(
    messages ?? []
  );

  const sendMessage = (newMessage: Message) => {
    setMessages((prevMessages) => [...prevMessages, newMessage]);
  };

  const sendRequest = async (newMessage: Message) => {
    sendMessage(newMessage);
    setInput((input) => newMessage.message);
    sendMessage({
      id: 1,
      avatar: '/User1.png',
      name: 'AI',
      message: "Extracting SQL schema from documents and adding them to your database...", // Or process result.data as needed
    });
    const result = await refetch();
    if (result.data) {
      sendMessage({
        id: 1,
        avatar: '/User1.png',
        name: 'AI',
        message: result.data.result, // Or process result.data as needed
      });
    }
  }

  return (
    <div className="flex flex-col justify-between w-full h-full">
      <ChatTopbar selectedUser={selectedUser} />

      <ChatList
        messages={messagesState}
        selectedUser={selectedUser}
        sendMessage={sendRequest}
        isMobile={isMobile}
      />
    </div>
  );
}
