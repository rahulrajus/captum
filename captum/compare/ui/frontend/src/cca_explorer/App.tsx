import React from "react";
import ExplorerContainer from "./components/ExplorerContainer";
import CKAExplorerContainer from "./components/CKAExplorerContainer";
import "./App.css";

interface AppConfig {
  sample_window_size: number,
  max_sample_window_size: number,
  min_sample_window_size: number,
}

interface AppState {
  config: AppConfig;
}

const initialState: AppState = {
  config: {
    sample_window_size: 5,
    max_sample_window_size: 8,
    min_sample_window_size: 3,
  },
};

export function App() {

  const [state, ] = React.useReducer(
    (prevState: AppState, newState: Partial<AppState>) => {
      return { ...prevState, ...newState };
    },
    initialState,
  );

  const body =
    (
      <div className="workspace-explorers">
        <ExplorerContainer
            key={0}
            idx={0}
            sampleWindowSize={state.config.sample_window_size}
          />
      </div>
    );

  const ckaBody =
    (
      <div className="workspace-explorers">
        {/* <ClusterContainer layerName="layer2.0.conv1" modelName="resnet32"/> */}
        <CKAExplorerContainer key={0} idx={0} sampleWindowSize={state.config.sample_window_size} />
      </div>
  )


  return (
    <div className="cca-app">
      <header className="cca-app-header">
        <div className="cca-app-header-title">Embedding Explorer</div>
      </header>
      {ckaBody}
    </div>
  );
}
