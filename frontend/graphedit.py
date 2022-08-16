import argparse
import copy
import os
import time
import json
import streamlit as st
import pandas as pd
import penman

from graphviz import Source
from xpotato.dataset.utils import default_pn_to_graph

from tuw_nlp.graph.utils import graph_to_pn, pn_to_graph

from utils import (
    train_df,
    add_rule_manually,
    annotate_df,
    extract_data_from_dataframe,
    get_df_from_rules,
    graph_viewer,
    init_evaluator,
    init_extractor,
    init_session_states,
    rank_and_suggest,
    read_df,
    rerun,
    rule_chooser,
    save_ruleset,
    read_ruleset,
    save_after_modify,
    save_dataframe,
    match_texts,
    show_ml_feature,
    st_stdout,
    to_dot,
)

def main():
    st.set_page_config(layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
	
                visibility: hidden;
                
                }
            footer:after {
                content:'GraphEdit - mx.markus.rei@gmx.net'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; color: black;'>GraphEdit</h1>"
        "<h2 style='text-align: center; color: black;'>Edit ud PENMAN graphs!</h2>",
        unsafe_allow_html=True,
    )
    init_session_states()
    text_input = st.text_area("Provide the text here you want to convert to Penman notation!"
    )
    topenman = st.button("To Penman!")
    output = st.empty()
    penman_input = st.text_area("Provide the penman notation here you want to render!"
    )
    global generatedpenman
    render = st.button("Render!")
    extractor = init_extractor("en", "ud")
    if topenman:
    	if text_input:
    		texts = text_input.split("\n")
    		graphs = list(extractor.parse_iterable([text for text in texts], "ud"))
    		dot_current_graph = to_dot(
    		graphs[0],
    		)
    		penmanstring = value=(penman.encode(penman.decode(graph_to_pn(graphs[0])), indent=10))
    		#penman_input = placeholder.text_area("Penman notation provided.", penmanstring)
    		output.text(penmanstring)
    		generatedpenman = penmanstring
    if render:
    	#penman_input = placeholder.text_area("Provide the penman notation here you want to render!", penmanstring)
    	if penman_input:

    		graph, ind = default_pn_to_graph(penman_input)
    		dot_current_graph = to_dot(
    		graph,
    		)

    		if st.session_state.download:
    			graph_pipe = Source(dot_current_graph).pipe(format="svg")
    			st.download_button(
    			label="Download graph as SVG",
    			data=graph_pipe,
    			file_name="graph.svg",
    			mime="mage/svg+xml",
    			)

    		with st.expander("Graph dot source", expanded=False):
    			st.write(dot_current_graph)

    		st.graphviz_chart(
    		dot_current_graph,
    		use_container_width=True,
    		)

    		st.write("Penman format:")
    		st.text(penman.encode(penman.decode(graph_to_pn(graph)), indent=10))
    		st.write("In one line format:")
    		st.write(graph_to_pn(graph))
    		
if __name__ == "__main__":
    main()
