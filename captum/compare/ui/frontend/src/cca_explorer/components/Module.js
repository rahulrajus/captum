import React from "react";
import { SamplesWindow } from "./Samples";
import "./Module.css";

function Component(props) {
  const windowSize = 3;
  let description = "";
  description = `CCA Direction ${props.id + 1}`;

  const onClick = (e) => props.onClick && props.onClick(e);
  const onViewDetailClick = () => {
    props.onViewDetail && props.onViewDetail();
  };
  let [samplesLo, samplesHi] = [null, null];

  const firstSampleIds = props.sampleIds.slice(0, windowSize);
  samplesLo = (
    <div className="module-component-samples">
      <SamplesWindow sampleIds={firstSampleIds} anchor="begin" />
    </div>
  );

  const lastSampleIds = props.sampleIds.slice(-windowSize);
  samplesHi = (
    <div className="module-component-samples">
      <SamplesWindow sampleIds={lastSampleIds} size={windowSize} anchor="end" />
    </div>
  );

  return (
    <div className="module-component-overview">
      <div className="module-component-overview-top">
        <div>{description}</div>
        <div
          className="module-component-view-detail"
          onClick={onViewDetailClick}
        >
          View Detail
        </div>
      </div>
      <div className="module-component-overview-bottom" onClick={onClick}>
        {samplesLo}
        {samplesHi}
      </div>
    </div>
  );
}

function Module(props) {
  const components = props.components;
  const componentsView = components.map((component, i) => {
    const onViewDetail = () => {
      props.onViewComponentDetail &&
        props.onViewComponentDetail({
          model_index: props.explorer_id,
          component_index: i,
        });
    };
    const onClick = (e) =>
      props.onComponentClick && props.onComponentClick(component, e);
    return (
      <Component
        id={i}
        sampleIds={component.ids}
        onViewDetail={onViewDetail}
        onClick={onClick}
      />
    );
  });

  return <div className="module">{componentsView}</div>;
}

export default Module;
