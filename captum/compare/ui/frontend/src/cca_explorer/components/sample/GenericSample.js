import React from "react";
import "./GenericSample.css";
import { LazyImage } from "../../../common/components";

const AttrType = Object.freeze({
  IMAGE: "image",
  AUDIO: "audio",
  VIDEO: "video",
});

function GenericSample(props) {
  const { sample, explorer } = props;
  const rows = [];
  const payload = sample.payload;
  for (const attrName in payload) {
    let value = payload[attrName];
    let attrContent;
    if (value?.type != null) {
      const attr = async () => {
        const val = await explorer.workspace.mbx_service.getGenericSampleAttr(
          explorer.workspace.id,
          explorer.id,
          sample.id,
          attrName,
        );
        return val;
      };
      switch (value.type) {
        case AttrType.IMAGE:
          attrContent = (
            <LazyImage
              className="sample-thumbnail"
              srcProvider={attr}
              height={200}
              width={200}
              threshold={[0.1, 0.5, 1]}
              rootMargin={"10px 10px 10px 10px"}
            />
          );
          break;
        case AttrType.VIDEO:
          attrContent = (
            <div>
              <video width="320" height="240" controls>
                <source src={attr} />
                your browser does not support html video tags
              </video>
            </div>
          );
          break;
        case AttrType.AUDIO:
          attrContent = (
            <div>
              <audio controls>
                <source src={attr} />
              </audio>
            </div>
          );
          break;
        default:
          attrContent = value.toString();
      }
    } else {
      attrContent = value?.toString();
    }
    const row = (
      <div key={attrName} className="generic-sample-row">
        <div className="generic-sample-attr-name">{attrName}</div>
        <div className="generic-sample-attr-content">{attrContent}</div>
      </div>
    );
    rows.push(row);
  }
  return <div className="generic-sample">{rows}</div>;
}

export default GenericSample;
