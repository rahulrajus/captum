import React from "react";
import { Samples } from "./Samples";
import "./ComponentDetail.css";

function ComponentDetail(props) {
  const component = props.component;
  let description = "";

  return (
    <div className="component-detail">
      <div className="component-detail-header">
        <div className="component-detail-module-name">{props.layer_name}</div>
      </div>
      <div className="component-detail-description">{description}</div>
      <Samples component={component} windowSize={props.sampleWindowSize} />
    </div>
  );
}

export default ComponentDetail;
