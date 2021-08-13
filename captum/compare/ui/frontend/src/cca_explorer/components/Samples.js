import React, { useState } from "react";
import SampleContainer from "./sample/SampleContainer";
import { Tooltip } from "./common/Tooltip";
import { LazyImage } from "../../common/components";
import { ReactComponent as SwapIcon } from "../icons/swap.svg";
import { getSampleImage } from "../services/api";
import "./Samples.css";

function Sample({ sampleId }) {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);

  const onTooltipMount = () => setIsTooltipVisible(true);
  const onTooltipHide = () => setIsTooltipVisible(false);

  const tooltipContent = isTooltipVisible ? (
    <div className="sample-tooltip-content">
      <SampleContainer sampleId={sampleId} />
    </div>
  ) : null;

  const img_url = async () => {
    const img_data = await getSampleImage(sampleId);
    return img_data;
  };

  return (
    <div className="sample-thumbnail-container">
      <Tooltip
        content={tooltipContent}
        placement="bottom"
        delayMs={100}
        trigger="mouseenter"
        onMount={onTooltipMount}
        onHide={onTooltipHide}
      >
        <div>
          <LazyImage
            className="sample-thumbnail"
            srcProvider={img_url}
            height={32}
            width={32}
            threshold={[0.1, 0.5, 1]}
            rootMargin={"10px 10px 10px 10px"}
          />
        </div>
      </Tooltip>
    </div>
  );
}

export function SamplesWindow({
  sampleIds,
  anchor = "center",
  columns = 3,
  size = 3,
  ...props
}) {
  const samples = sampleIds.map((sampleId, i) => {
    const onSampleClick = () =>
      props.onSampleClick && props.onSampleClick(sampleId);
    return <Sample key={i} sampleId={sampleId} onClick={onSampleClick} />;
  });
  columns = Math.min(columns, sampleIds.length);
  const style = {
    gridTemplateColumns: `repeat(${columns}, 1fr)`,
  };
  return (
    <div className="samples-window-grid" style={style}>
      {samples}
    </div>
  );
}

const middleWindow = (sampleIds, numValues) => {
  const middle_value = Math.round(sampleIds.length / 2);
  let leftIdx = middle_value;
  let rightIdx = middle_value;
  while (leftIdx >= 0 && rightIdx < sampleIds.length) {
    if (leftIdx > 0) {
      leftIdx--;
    }
    if (rightIdx < sampleIds.length) {
      rightIdx++;
    }
    if (rightIdx - leftIdx + 1 >= numValues) {
      break;
    }
  }
  return sampleIds.slice(leftIdx, rightIdx);
};

export function Samples({ windowSize = 3, component, ...props }) {
  const [focusMid, setFocusMid] = useState(false);

  if (component == null)
    throw new Error("component object required to display samples");

  const numSampleBins = component.length;

  if (numSampleBins === 0) return null;

  const focusedSize = windowSize * windowSize;
  const focusedColumns = windowSize;
  const unfocusedSize = 4;
  const unfocusedColumns = 2;

  let windowSizeMid, windowSizeLo, columnsMid, columnsLo;

  if (focusMid) {
    windowSizeMid = focusedSize;
    columnsMid = focusedColumns;
    windowSizeLo = unfocusedSize;
    columnsLo = unfocusedColumns;
  } else {
    windowSizeMid = unfocusedSize;
    columnsMid = unfocusedColumns;
    windowSizeLo = focusedSize;
    columnsLo = focusedColumns;
  }

  const firstSampleBin = component.ids.slice(0, windowSizeLo);
  const lastSampleBin = component.ids.slice(-windowSizeLo);
  const midSampleBin = middleWindow(component.ids, windowSizeMid);

  const windowSizeHi = windowSizeLo;
  const columnsHi = columnsLo;

  const onSampleClick = (sampleId) =>
    props.onSampleClick && props.onSampleClick(sampleId);
  const samplesLo = (
    <SamplesWindow
      sampleIds={firstSampleBin}
      anchor="begin"
      onSampleClick={onSampleClick}
      size={windowSizeLo}
      columns={columnsLo}
    />
  );
  const samplesHi = (
    <SamplesWindow
      sampleIds={lastSampleBin}
      anchor="end"
      onSampleClick={onSampleClick}
      size={windowSizeHi}
      columns={columnsHi}
    />
  );
  const samplesMid = (
    <SamplesWindow
      sampleIds={midSampleBin}
      anchor="end"
      onSampleClick={onSampleClick}
      size={windowSizeMid}
      columns={columnsMid}
    />
  );

  const onSwapClick = () => setFocusMid(!focusMid);

  const btnSwap = (
    <button onClick={onSwapClick} className="btn-icon">
      <SwapIcon />
    </button>
  );

  return (
    <div className="samples-container">
      <div className="samples-container-bottom">
        {samplesLo}
        <div className="samples-container-bottom-mid">
          <div className="samples-mid">{samplesMid}</div>
          {btnSwap}
        </div>
        {samplesHi}
      </div>
    </div>
  );
}

export default Sample;
