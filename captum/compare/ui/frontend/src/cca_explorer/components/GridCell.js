import React, { useState } from "react";
import colormap from "colormap";
import "./GridCell.css";

const colors = colormap({
  colormap: "magma",
  nshades: 100,
  format: "hex",
  alpha: 1,
});

function GridCell(props) {
  const [hovered, setHovered] = useState(false);
  let cca_score = props.score;
  if (!cca_score) {
    cca_score = 0.0;
  }
  let color_index = Math.round(cca_score * 100);
  color_index = Math.min(color_index, 99);
  color_index = Math.max(0, color_index);
  console.log(colors[color_index]);

  function darkenColor(hexInput, percent) {
    let hex = hexInput;

    // strip the leading # if it's there
    hex = hex.replace(/^\s*#|\s*$/g, "");

    // convert 3 char codes --> 6, e.g. `E0F` --> `EE00FF`
    if (hex.length === 3) {
      hex = hex.replace(/(.)/g, "$1$1");
    }

    let r = parseInt(hex.substr(0, 2), 16);
    let g = parseInt(hex.substr(2, 2), 16);
    let b = parseInt(hex.substr(4, 2), 16);

    const calculatedPercent = (100 + percent) / 100;

    r = Math.round(Math.min(255, Math.max(0, r * calculatedPercent)));
    g = Math.round(Math.min(255, Math.max(0, g * calculatedPercent)));
    b = Math.round(Math.min(255, Math.max(0, b * calculatedPercent)));

    return `#${r.toString(16).toUpperCase()}${g.toString(16).toUpperCase()}${b
      .toString(16)
      .toUpperCase()}`;
  }
  let style = { background: colors[color_index] };

  if (hovered) {
    style["background"] = darkenColor(style["background"].substring(1), -10);
    style["border"] = "1px solid black";
  }

  return (
    <div
      onMouseEnter={() => {
        props.setLabels([props.r, props.c]);
        setHovered(true);
      }}
      onMouseLeave={() => {
        props.setLabels(null);
        setHovered(false);
      }}
      className="grid-cell-container"
      style={style}
    >
      {Math.round(props.score * 100)}
    </div>
  );
}

export default GridCell;
