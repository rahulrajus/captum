import React from "react";
import { App as CCAExplorer } from "./cca_explorer/App";
import { CaptumLogo } from "./common/icons";
import "./App.css";

export function AppWrapper(props: React.PropsWithChildren<{}>) {
  return (
    <div className="app-container">
      <div id="app-header">
        <CaptumLogo id="app-header--logo" />
        <h1>Captum Insights</h1>
      </div>
      <div id="app-content">{props.children}</div>
    </div>
  );
}

export function App() {

  const LensRoot = () => {
    return <CCAExplorer />;
  };

  return (
    <AppWrapper>
      <LensRoot />
    </AppWrapper>
  );
}
