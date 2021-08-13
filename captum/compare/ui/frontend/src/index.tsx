import React from "react";
import { render } from "react-dom";
import { App } from "./App";

import "./index.css";

render(
  <div className="lenses-webapp">
    <App/>
  </div>,
  document.getElementById("root")
);
