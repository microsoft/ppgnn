# PPGNN: A Piece-Wise Polynomial Filtering Approach for Graph Neural Networks

This repo contains the code for the paper published at `ECML PKDD 2022: Machine
Learning and Knowledge Discovery in Databases`. The link to our ECML publication
can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-26390-3_25)
and the link to the paper on arXiv can be found
[here](https://arxiv.org/abs/2112.03499).

## Citation

If you find this work useful, please cite our paper. Note that the first three authors contributed equally to this work.

```bibtex
@InProceedings{10.1007/978-3-031-26390-3_25,
author="Lingam, Vijay
and Sharma, Manan
and Ekbote, Chanakya
and Ragesh, Rahul
and Iyer, Arun
and Sellamanickam, Sundararajan",
editor="Amini, Massih-Reza
and Canu, St{\'e}phane
and Fischer, Asja
and Guns, Tias
and Kralj Novak, Petra
and Tsoumakas, Grigorios",
title="A Piece-Wise Polynomial Filtering Approach for Graph Neural Networks",
booktitle="Machine Learning and Knowledge Discovery in Databases",
year="2023",
publisher="Springer International Publishing",
address="Cham",
pages="412--452",
abstract="Graph Neural Networks (GNNs) exploit signals from node features and the input graph topology to improve node classification task performance. Recently proposed GNNs work across a variety of homophilic and heterophilic graphs. Among these, models relying on polynomial graph filters have shown promise. We observe that polynomial filter models need to learn a reasonably high degree polynomials without facing any over-smoothing effects. We find that existing methods, due to their designs, either have limited efficacy or can be enhanced further. We present a spectral method to learn a bank of filters using a piece-wise polynomial approach, where each filter acts on a different subsets of the eigen spectrum. The approach requires eigendecomposition only for a few eigenvalues at extremes (i.e., low and high ends of the spectrum) and offers flexibility to learn sharper and complex shaped frequency responses with low-degree polynomials. We theoretically and empirically show that our proposed model learns a better filter, thereby improving classification accuracy. Our model achieves performance gains of up to {\$}{\$}{\backslash}sim {\$}{\$}∼6{\%} over the state-of-the-art (SOTA) models while being only {\$}{\$}{\backslash}sim {\$}{\$}∼2x slower than the recent spectral approaches on graphs of sizes up to {\$}{\$}{\backslash}sim {\$}{\$}∼169K nodes.",
isbn="978-3-031-26390-3"
}
```

## Abstract

Graph Neural Networks (GNNs) exploit signals from node features and the input
graph topology to improve node classification task performance. However, these
models tend to perform poorly on heterophilic graphs, where connected nodes have
different labels. Recently proposed GNNs work across graphs having varying levels
of homophily. Among these, models relying on polynomial graph filters have shown
promise. We observe that solutions to these polynomial graph filter models are
also solutions to an overdetermined system of equations. It suggests that in
some instances, the model needs to learn a reasonably high order polynomial.
On investigation, we find the proposed models ineffective at learning such
polynomials due to their designs. To mitigate this issue, we perform an
eigendecomposition of the graph and propose to learn multiple adaptive
polynomial filters acting on different subsets of the spectrum. We theoretically
and empirically show that our proposed model learns a better filter, thereby
improving classification accuracy. We study various aspects of our proposed
model including, dependency on the number of eigencomponents utilized, latent
polynomial filters learned, and performance of the individual polynomials on
the node classification task. We further show that our model is scalable by
evaluating over large graphs. Our model achieves performance gains of up to 5%
over the state-of-the-art models and outperforms existing polynomial
filter-based approaches in general.

## Steps To Run

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Download the public datasets present in an drive link:

```bash
cd ppgnn
bash scripts/download_data.sh
```

3. Run the bash scripts for a particular dataset:

```bash
cd ppgnn/src
bash ../scripts/run_<dataset>.sh

#For example:

cd ppgnn/src
bash ../scripts/run_cora.sh
```

## People Involved

This project would not have been possible without the involvement of the following people:

- Vijay Lingam [📧](mailto:vijaylingam0810@gmail.com)
- Manan Sharma [📧](mailto:manan2908@gmail.com)
- Chanakya Ekbote [📧](mailto:chanakyekbote@gmail.com)
- Rahul Ragesh [📧](mailto:rahulragesh@microsoft.com)
- Arun Iyer [📧](mailto:ariy@microsoft.com)
- Sundararajan Sellamanickam [📧](mailto:ssrajan@microsoft.com)
- B. Ashok (BASH) [📧](mailto:bash@microsoft.com)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
