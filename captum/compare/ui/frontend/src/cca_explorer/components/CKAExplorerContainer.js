import React, { useState, useEffect } from "react";
import Spinner from "./common/Spinner";
import Modules from "./Modules";
import ClusterContainer from "./ClusterContainer"
import Card from "./common/Card";

import "./ExplorerContainer.css";
import axios from "axios";

function CKAExplorerContainer(props) {
  const [layer1Selection, setLayer1Selection] = useState(0);
  const [model, setModel] = useState({});

  useEffect(() => {
    async function getData() {
      const resp = await axios.get("/info");
      setModel(resp.data);
    }
    getData();
  }, []);

  const onLayer1ModuleListModuleClick = (id) => {
    setLayer1Selection(id);
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
      </div>
    );

    body = (
      <div className="explorer-body">
        <div className="explorer-modules">
          {left}
          <ClusterContainer layerName={model.model1?.layers[layer1Selection]} modelName={model.model1?.name}/>
        </div>
      </div>
    );
  }

  return (
    <div className="explorer-container">
      <div className="explorer-header">
        {`CKA Explorer ${model.model1?.name}`}
      </div>
      {body}
    </div>
  );
}

export default CKAExplorerContainer;
