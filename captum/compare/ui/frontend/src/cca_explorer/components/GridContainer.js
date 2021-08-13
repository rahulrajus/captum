import React, { useState, useEffect } from "react";
// import { ReactComponent as LeftArrow } from "../icons/leftArrow.svg";
import GridCell from "./GridCell";
import axios from "axios";

function GridContainer(props) {
  const [grid, setGrid] = useState([[]]);
  const [pairSelected, setPairSelected] = useState();
  const layer1_names = props.layer1_names ? props.layer1_names : [];
  const layer2_names = props.layer2_names ? props.layer1_names : [];
  useEffect(() => {
    async function getData() {
      const resp = await axios.get("/similarities");
      setGrid(resp.data.data);
    }
    getData();
  }, []);

  const setLabels = (pair) => {
    setPairSelected(pair);
  };

  const column_size = "100px";
  const gray_label = "#808080";
  const black_label = "#000000";

  let grid_columns = [];
  for (let i = 0; i < grid.length + 1; i++) {
    grid_columns.push(column_size);
  }
  let grid_template_columns = `${grid_columns.join(" ")}`;
  let cells = [];
  cells.push(<div style={{ height: "20px" }}></div>);
  const layer2_titles = layer2_names.map((title) => {
    return (
      <div
        style={{
          height: "20px",
          color:
            pairSelected && title === pairSelected[1]
              ? black_label
              : gray_label,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {title}{" "}
      </div>
    );
  });
  cells.push(layer2_titles);

  for (let row = 0; row < grid.length; row++) {
    cells.push(
      <div
        style={{
          height: "100px",
          color:
            pairSelected && row === pairSelected[0] ? black_label : gray_label,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {layer1_names[row]}
      </div>,
    );
    for (let col = 0; col < grid[0].length; col++) {
      cells.push(
        <GridCell
          score={grid[row][col]}
          r={row}
          c={col}
          setLabels={setLabels}
        />,
      );
    }
  }

  return (
    <div
      style={{
        margin: "10px auto",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: grid_template_columns,
          columnGap: "5px",
          rowGap: "5px",
          height: "100%",
        }}
      >
        {cells}
      </div>
    </div>
  );
}

export default GridContainer;
