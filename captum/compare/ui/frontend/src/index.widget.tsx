import React from "react";
import ReactDOM from "react-dom";
import { App } from "./App";

import "./index.widget.css";

export function create(messengerInfo: { target: string; type: string }) {
  const widgetContainer = document.createElement("div");
  widgetContainer.classList.add("lenses-widget");
  ReactDOM.render(<App />, widgetContainer);
  return widgetContainer;
}
