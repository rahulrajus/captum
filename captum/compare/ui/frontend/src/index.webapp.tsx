import React from "react";
import ReactDOM from "react-dom";
import { App } from "./App";

import "./index.css";

export function create(messengerInfo: { target: string; type: string }) {
  const appContainer = document.createElement("div");
  appContainer.classList.add("lenses-webapp");
  ReactDOM.render(<App />, appContainer);
  return appContainer;
}
