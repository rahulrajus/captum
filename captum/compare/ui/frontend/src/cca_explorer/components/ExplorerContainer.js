import React, { useState, useEffect } from "react";
import { ReactComponent as LeftArrow } from "../icons/leftArrow.svg";
import Spinner from "./common/Spinner";
import Modules from "./Modules";
import GridContainer from "./GridContainer";
import ComponentDetail from "./ComponentDetail";
import Card from "./common/Card";
import Module from "./Module";
import cx from "../utils/cx";

import "./ExplorerContainer.css";
import axios from "axios";

function TabHeader(props) {
  return (
    <div className="explorer-tab-view-header">
      <div
        className={cx(["explorer-tab-view-header-tab"])}
        onClick={() =>
          props.onActiveTabUpdate && props.onActiveTabUpdate(false)
        }
      >
        Samples
      </div>
    </div>
  );
}

function ExplorerContainer(props) {
  const [layer1Selection, setLayer1Selection] = useState(0);
  const [layer2Selection, setLayer2Selection] = useState(0);
  const [componentData, setComponentData] = useState({
    layer1_dimensions: [],
    layer2_dimensions: [],
    similarity: 0.0,
  });

  const [detailClicked, setDetailClicked] = useState({});
  const [model, setModel] = useState({});

  async function getComponentData(layer1, layer2) {
    const resp = await axios.get(`/compare/${layer1}/${layer2}`);
    setComponentData(resp.data.data);
  }

  useEffect(() => {
    async function getData() {
      const resp = await axios.get("/info");
      setModel(resp.data);
      getComponentData(
        resp.data.model1?.layers[0],
        resp.data.model2?.layers[0],
      );
    }
    getData();
  }, []);

  const maxSampleWindowSize = props.maxSampleWindowSize ?? 6;
  const minSampleWindowSize = props.minSampleWindowSize ?? 3;

  const initialSampleWindowSize =
    props.sampleWindowSize == null
      ? 3
      : Math.min(
          Math.max(props.sampleWindowSize, minSampleWindowSize),
          maxSampleWindowSize,
        );
  const [sampleWindowSize, setSampleWindowSize] = useState(
    initialSampleWindowSize,
  );

  const onComponentClick = (component, e) =>
    props.onComponentClick && props.onComponentClick(component, e);

  const onViewComponentDetail = (component) => {
    setDetailClicked(component);
  };

  const onLayer1ModuleListModuleClick = (id) => {
    setLayer1Selection(id);
    getComponentData(
      model.model1?.layers[id],
      model.model2?.layers[layer2Selection],
    );
  };
  const onLayer2ModuleListModuleClick = (id) => {
    setLayer2Selection(id);
    getComponentData(
      model.model1?.layers[layer1Selection],
      model.model2?.layers[id],
    );
  };

  let body;
  if (model.model1?.layers == null || model.model2?.layers == null) {
    body = <Spinner />;
  } else {
    const left = (
      <div className="explorer-body-left">
        <Card
          title={`${model.model1?.name} Layers (${model.model1?.layers.length})`}
        >
          <Modules
            model={model.model1}
            selectedId={layer1Selection}
            direction="column"
            componentDirection="row"
            onModuleClick={onLayer1ModuleListModuleClick}
            componentMode={props.componentMode}
            updateColor={false}
          />
        </Card>
        <Card
          title={`${model.model2?.name} Layers (${model.model2?.layers.length})`}
        >
          <Modules
            model={model.model2}
            selectedId={layer2Selection}
            direction="column"
            componentDirection="row"
            onModuleClick={onLayer2ModuleListModuleClick}
            componentMode={props.componentMode}
            updateColor={false}
          />
        </Card>
      </div>
    );

    let rightContent;
    let rightTitleText;
    let tabContent;

    const tabHeader = <TabHeader />;

    if (detailClicked.model_index == null) {
      rightTitleText = `CCA Comparison`;
    } else {
      const model_name = model[`model${detailClicked.model_index + 1}`].name;
      let layer_name = model.model1?.layers[layer1Selection];
      if (detailClicked.model_index === 1) {
        layer_name = model.model2?.layers[layer2Selection];
      }
      rightTitleText = `${model_name}/${layer_name}/CCA Direction ${detailClicked.component_index}`;
    }

    if (detailClicked.model_index != null) {
      const layer_key = `layer${detailClicked.model_index + 1}_dimensions`;
      let layer_name = model.model1?.layers[layer1Selection];
      if (detailClicked.model_index === 1) {
        layer_name = model.model2?.layers[layer2Selection];
      }
      tabContent = (
        <ComponentDetail
          sampleWindowSize={sampleWindowSize}
          component={componentData[layer_key][detailClicked.component_index]}
          model_info={model.model1}
          layer_name={layer_name}
        />
      );
    } else {
      tabContent = (
        <div>
          <div>
            <center>
              <b>CCA Mean Similarity:</b> {componentData.similarity}
            </center>
          </div>
          <div style={{ width: "50%", float: "left" }}>
            <Card
              title={`${model.model1?.name} (${model.model1?.layers[layer1Selection]})`}
            >
              <Module
                key={'model1-dimensions'}
                explorer_id={0}
                components={componentData.layer1_dimensions}
                onViewComponentDetail={onViewComponentDetail}
                onComponentClick={onComponentClick}
              />
            </Card>
          </div>
          <div style={{ width: "50%", float: "right" }}>
            <Card
              title={`${model.model2?.name} (${model.model2?.layers[layer2Selection]})`}
            >
              <Module
                key={'model2-dimensions'}
                explorer_id={1}
                components={componentData.layer2_dimensions}
                onViewComponentDetail={onViewComponentDetail}
                onComponentClick={onComponentClick}
              />
            </Card>
          </div>
        </div>
      );
    }

    rightContent = (
      <div className="explorer-tab-view">
        {tabHeader}
        <div className="explorer-tab-view-body">{tabContent}</div>
      </div>
    );

    const onBackClick = () => setDetailClicked({});

    const btnBack = (
      <button
        className={cx(["btn-icon", "btn-back"])}
        onClick={onBackClick}
        disabled={!detailClicked}
      >
        <LeftArrow />
      </button>
    );

    const onMoreSamplesClick = () =>
      setSampleWindowSize(Math.min(sampleWindowSize + 1, maxSampleWindowSize));

    const onFewerSamplesClick = () =>
      setSampleWindowSize(Math.max(sampleWindowSize - 1, minSampleWindowSize));

    const btnMoreSamples = (
      <button
        className={cx(["btn-icon", "btn-more-samples"])}
        disabled={sampleWindowSize === maxSampleWindowSize}
        onClick={onMoreSamplesClick}
      >
        +
      </button>
    );

    const btnFewerSamples = (
      <button
        className={cx(["btn-icon", "btn-fewer-samples"])}
        disabled={sampleWindowSize === minSampleWindowSize}
        onClick={onFewerSamplesClick}
      >
        -
      </button>
    );

    const rightTitle = (
      <div className="explorer-detail-title">
        {btnBack}
        <div>{rightTitleText}</div>
        <div
          className="explorer-detail-title-btn-group"
          style={{
            visibility:
              detailClicked.model_index != null ? "visible" : "hidden",
          }}
        >
          {btnFewerSamples}
          {btnMoreSamples}
        </div>
      </div>
    );
    const right = (
      <div className="explorer-body-right">
        <Card title={rightTitle}>{rightContent}</Card>
      </div>
    );

    body = (
      <div className="explorer-body">
        <div>
          <Card title={"Layer Comparisons"}>
            <GridContainer
              layer1_names={model.model1?.layers}
              layer2_names={model.model2?.layers}
            />
          </Card>
        </div>
        <div className="explorer-modules">
          {left}
          {right}
        </div>
      </div>
    );
  }

  return (
    <div className="explorer-container">
      <div className="explorer-header">
        {`Explorer ${model.model1?.name} vs. ${model.model2?.name}`}
      </div>
      {body}
    </div>
  );
}

export default ExplorerContainer;
