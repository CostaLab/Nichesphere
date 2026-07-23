import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_custom_sankey(adata, table_key='TPO_x_EV', filter_df=None, 
                       source_cell=None, ligand_gene=None, 
                       receptor_gene=None, target_cell=None,
                       ligand_list=None, receptor_list=None,
                       threshold=25, title="Parallel Flow Map",
                       save_path=None):
    """Flat 2D Sankey diagram of ligand-receptor interactions from pyCrossTalkeR results.

    Visualises the flow of differential LR communication scores across four hierarchical
    layers — source cell type → ligand → receptor → target cell type — as a flat,
    R-style Sankey diagram built with Plotly.  Links are drawn with uniform widths so
    that the diagram emphasises *which* interactions are active rather than their
    relative magnitudes; actual ``LRScore`` values are encoded by link colour (positive
    scores in soft red, negative in soft blue) and exposed on hover, with a diverging
    colour bar for reference.

    All optional filters are combined with strict AND logic: only interactions that
    simultaneously satisfy every supplied criterion are shown.  Interactions are ranked
    by absolute ``LRScore`` and capped at ``threshold`` before plotting.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing pyCrossTalkeR output in
        ``adata.uns['pycrosstalker']['results']['tables']``.
    table_key : str, default ``'TPO_x_EV'``
        Key of the comparison table to retrieve from the pyCrossTalkeR results, typically
        in the format ``'conditionA_x_conditionB'``. The colour bar title is derived
        automatically from this key (e.g. ``'TPO_x_EV'`` → ``'LRScore (TPO vs EV)'``).
    filter_df : pd.DataFrame or None, default None
        Optional pre-computed filter table with a boolean ``'filter'`` column (1 = keep).
        Its index must match ``'source->target'`` strings.  Only interactions whose
        ``source->target`` key appears in ``filter_df`` with ``filter == 1`` are retained
        before any other filtering step is applied.
    source_cell : str or list of str or None, default None
        Source cell type(s) to include.  Accepts a single string or a list; if ``None``,
        all source cell types are kept.
    ligand_gene : str or None, default None
        Substring filter applied to ``gene_A`` (case-insensitive).  Useful for quickly
        finding a specific ligand by partial name.
    receptor_gene : str or None, default None
        Substring filter applied to ``gene_B`` (case-insensitive).  Useful for quickly
        finding a specific receptor by partial name.
    target_cell : str or list of str or None, default None
        Target cell type(s) to include.  Accepts a single string or a list; if ``None``,
        all target cell types are kept.
    ligand_list : list of str or None, default None
        Exact-match whitelist of ligand gene names (e.g. a biological process gene set
        such as ECM glycoproteins).  The ``|L`` suffix produced by pyCrossTalkeR is
        stripped before matching, and comparison is case-insensitive.  Takes precedence
        over ``ligand_gene`` when both are provided (both filters are applied).
    receptor_list : list of str or None, default None
        Exact-match whitelist of receptor gene names.  The ``|R`` suffix is stripped
        before matching; comparison is case-insensitive.
    threshold : int, default 25
        Maximum number of interactions to display.  After all filters are applied,
        interactions are sorted by absolute ``LRScore`` (descending) and only the top
        ``threshold`` are plotted.
    title : str, default ``'Parallel Flow Map'``
        Title displayed above the Sankey diagram and used as the default filename stem
        when saving.
    save_path : str or None, default None
        If provided, the figure is saved as a self-contained HTML file (the extension
        is replaced with ``.html`` regardless of what is passed).  The HTML embeds a
        Plotly toolbar configured to export SVG, and a JavaScript snippet that
        automatically triggers the download button one second after the page loads.
        The file is then opened in the default web browser.  Check the browser's
        downloads directory for the resulting ``.svg`` file.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The Plotly figure object, which can be displayed with ``fig.show()`` or further
        customised before rendering.  Returns ``None`` if no interactions match the
        specified filters.

    Raises
    ------
    KeyError
        If ``adata.uns`` does not contain a ``'pycrosstalker'`` key.

    Notes
    -----
    * Link widths are deliberately uniform (all set to 1) so the layout emphasises
      the *topology* of active interactions rather than their magnitude.  The actual
      ``LRScore`` is available on hover and via the colour bar.
    * The four-layer node structure appends role suffixes (``(Src)``, ``(Lig)``,
      ``(Rec)``, ``(Tgt)``) internally to disambiguate nodes that share a name across
      layers (e.g. a gene that appears as both ligand and receptor); these suffixes are
      stripped from the visible node labels in the final plot.
    * Saving relies on the browser's built-in download mechanism rather than
      ``kaleido`` or ``orca``, making it dependency-free for vector export.

    Examples
    --------
    Basic usage — plot all interactions from a comparison table:

    >>> fig = nichesphere.comm.plot_custom_sankey(adata, table_key='TPO_x_EV')
    >>> fig.show()

    Filter to a specific source cell type and a gene-set list:

    >>> ecm_genes = ['FN1', 'COL1A1', 'THBS1', 'VTN']
    >>> fig = nichesphere.comm.plot_custom_sankey(
    ...     adata,
    ...     table_key='TPO_x_EV',
    ...     source_cell='Fibroblast',
    ...     ligand_list=ecm_genes,
    ...     threshold=20,
    ...     title='ECM Glycoprotein Signalling – TPO vs EV',
    ...     save_path='figures/ecm_sankey.html',
    ... )
    """
    
    # 1. Pull the comparative data table
    if 'pycrosstalker' not in adata.uns:
        raise KeyError("pyCrossTalkeR data not found in adata.uns")
    
    data_sub = adata.uns['pycrosstalker']['results']['tables'][table_key].copy()
    
    # 2. Apply your specific logical index slice if a filter dataframe is passed
    if filter_df is not None:
        sig_pairs_arrow = filter_df.index[filter_df['filter'] == 1]
        data_sub = data_sub.loc[[
            j in sig_pairs_arrow
            for j in data_sub.source + '->' + data_sub.target
        ]]
    
    # 3. STRICT INTERSECTION (AND) THRESHOLD FILTERING
    mask = pd.Series(True, index=data_sub.index)
    
    if source_cell is not None:
        src_list = [source_cell] if isinstance(source_cell, str) else list(source_cell)
        mask = mask & data_sub['source'].isin(src_list)
        
    if target_cell is not None:
        tgt_list = [target_cell] if isinstance(target_cell, str) else list(target_cell)
        mask = mask & data_sub['target'].isin(tgt_list)
        
    if ligand_gene is not None:
        mask = mask & data_sub['gene_A'].str.contains(ligand_gene, case=False, na=False)
        
    if receptor_gene is not None:
        mask = mask & data_sub['gene_B'].str.contains(receptor_gene, case=False, na=False)
        
    # --- Filter by a predefined list of Ligand Genes (like ECM_GPs) ---
    if ligand_list is not None:
        clean_lig_list = [str(x).strip().lower() for x in ligand_list]
        clean_gene_A = data_sub['gene_A'].str.replace('|L', '', regex=False).str.strip().str.lower()
        mask = mask & clean_gene_A.isin(clean_lig_list)
        
    # --- Filter by a predefined list of Receptor Genes (Optional) ---
    if receptor_list is not None:
        clean_rec_list = [str(x).strip().lower() for x in receptor_list]
        clean_gene_B = data_sub['gene_B'].str.replace('|R', '', regex=False).str.strip().str.lower()
        mask = mask & clean_gene_B.isin(clean_rec_list)
        
    # Slice the finalized intersected dataframe
    plot_df = data_sub[mask].copy()
    
    if plot_df.empty:
        print("No interactions found matching ALL specified cell and gene criteria simultaneously.")
        return None
        
    # Sort by absolute score value and apply the threshold cap
    plot_df['abs_score'] = plot_df['LRScore'].abs()
    plot_df = plot_df.sort_values(by='abs_score', ascending=False)
    
    if len(plot_df) > threshold:
        plot_df = plot_df.head(threshold)
        
    print(f"Plotting {len(plot_df)} distinct fluid flow paths.")

    # 4. RECONSTRUCT HIERARCHICAL LAYER CHANNELING
    plot_df['src_node'] = plot_df['source'] + " (Src)"
    plot_df['lig_node'] = plot_df['gene_A'] + " (Lig)"
    plot_df['rec_node'] = plot_df['gene_B'] + " (Rec)"
    plot_df['tgt_node'] = plot_df['target'] + " (Tgt)"

    edges1 = plot_df[['src_node', 'lig_node', 'LRScore']].values
    edges2 = plot_df[['lig_node', 'rec_node', 'LRScore']].values
    edges3 = plot_df[['rec_node', 'tgt_node', 'LRScore']].values
    all_edges = np.vstack([edges1, edges2, edges3])

    # 5. MAP TO TEXT LABELS & INDEXES SAFELY
    unique_nodes = list(np.unique(all_edges[:, :2]))
    node_dict = {name: idx for idx, name in enumerate(unique_nodes)}

    sources = [node_dict[row[0]] for row in all_edges] 
    targets = [node_dict[row[1]] for row in all_edges] 
    actual_scores = all_edges[:, 2].astype(float)

    # 6. ENFORCE UNIFORM LINE WIDTHS & DYNAMIC COLOR MAPPING
    link_colors = []
    equal_widths = np.ones(len(actual_scores)) 

    for score in actual_scores:
        if score > 0:
            link_colors.append("rgba(239, 138, 98, 0.5)")  # Soft Red
        else:
            link_colors.append("rgba(103, 169, 207, 0.5)") # Soft Blue

    # 7. BUILD THE FLAT 2D CONSTANT-WIDTH SANKEY
    fig = go.Figure()

    fig.add_trace(go.Sankey(
        arrangement = "snap", 
        node = dict(
            pad = 35,          
            thickness = 12,
            line = dict(color = "black", width = 1.0), 
            label = [name.split(" (")[0] for name in unique_nodes], 
            color = "lightgray"
        ),
        link = dict(
            source = sources,
            target = targets,
            value = equal_widths, 
            line = dict(color = "rgba(0, 0, 0, 0.6)", width = 0.8), 
            color = link_colors,
            customdata = actual_scores,
            hovertemplate = 'Flow Value (LRScore): %{customdata:.3f}<extra></extra>'
        )
    ))

    # Add the color bar trace
    colorbar_label = "LRScore<br>(" + table_key.replace("_x_", " vs ") + ")"
    max_val = float(np.max(np.abs(actual_scores))) if len(actual_scores) > 0 else 1.0
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='RdBu_r', 
            cmin=-max_val,
            cmax=max_val,
            colorbar=dict(
                title=dict(text=colorbar_label, side="top"),
                thickness=15,
                len=0.6, 
                x=1.12   
            ),
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))

    # FIXED: Clean update layout with no invalid 'config' keys
    fig.update_layout(
        title_text=title, 
        font_size=12, 
        width=900, 
        height=600,
        margin=dict(r=160),
        plot_bgcolor="white",  
        paper_bgcolor="white",  
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # --- AUTOMATED VECTOR SAVING BACKEND WORKAROUND ---
    if save_path is not None:
        import os
        import webbrowser
        
        # Ensure name points to a distinct html layout file name
        base_name = os.path.splitext(save_path)[0]
        html_path = base_name + ".html"
        
        # FIXED: Pass the image option configs directly into the to_html function where it's valid
        image_config = {
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': title.lower().replace(' ', '_'),
                'height': 600,
                'width': 900,
                'scale': 2
            }
        }
        
        html_string = fig.to_html(include_plotlyjs='cdn', config=image_config)
        
        # Injects script instruction forcing the web canvas layer to click the download icon automatically
        auto_download_script = """
        <script>
        window.onload = function() {
            setTimeout(function() {
                var camera_btn = document.querySelector('[data-title="Download plot as a png"]') || document.querySelector('[data-title="Download plot as a svg"]');
                if (camera_btn) { camera_btn.click(); }
            }, 1000);
        };
        </script>
        """
        
        with open(html_path, "w") as f:
            f.write(html_string + auto_download_script)
            
        print(f"Opening browser instance to trigger automated vector extraction. Check your computer downloads directory for your file.")
        webbrowser.open('file://' + os.path.realpath(html_path))

    return fig
