import React, { useState, useEffect } from "react";
import Spinner from "../common/Spinner";
import GenericSample from "./GenericSample";
import { getSampleImage } from "../../services/api";
import "./SampleContainer.css";

function SampleContainer(props) {
  const sampleId = props.sampleId;

  const [sample, setSample] = useState(null);

  useEffect(() => {
    (async () => {
      const img_info = await getSampleImage(sampleId);
      setSample(img_info);
    })();
  }, [sampleId]);

  const content =
    sample == null ? <Spinner /> : <GenericSample sample={sample} />;
  return <div className="sample-container">{content}</div>;
}

export default SampleContainer;
