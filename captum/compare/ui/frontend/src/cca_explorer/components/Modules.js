import React from "react";
import { RGBColor } from "../utils/color";
import "./Modules.css";

function Modules(props) {
  const model = props.model;

  const names = model.layers;

  const rows = [];
  const padding = props.padding == null ? "10px" : props.padding;

  for (let i = 0; i < names.length; i++) {
    const onModuleClick = () => {
      props.onModuleClick(i);
    };
    let cells = [];
    let cellStyle = {};
    if (props.selectedId === i)
      cellStyle.backgroundColor = RGBColor.lightOrange.toString();
    cells.push(
      <div key={names[i]} className="table-cell-display" style={cellStyle}>
        <div style={{ padding: padding }}>{names[i]}</div>
      </div>,
    );

    rows.push(
      <div key={i} className="table-row-display" onClick={onModuleClick}>
        {cells}
      </div>,
    );
  }

  return <div className="table-display">{rows}</div>;
}

export default Modules;
