FROM jupyter/tensorflow-notebook

RUN conda install -c pytorch --quiet --yes pytorch && \
    conda install -c fastai --quiet --yes fastai && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
