# Distributed Quantum Computing workflows with ColonyOS

Quantum computing is evolving rapidly across the globe, and Europe is no exception, with significant investments powering new research frontiers. However, a key challenge lies in efficiently orchestrating complex computations that often require integrating classical high-performance computing (HPC) resources with diverse quantum hardware. Here is an overview on [ColonyOS](https://colonyos.io/), an open-source meta-operating system designed to address this need by orchestrating distributed computing workflows across heterogeneous environments (including cloud, HPC, standalone workstations, and can be extended towards quantum systems). The following sections discuss the benefits of such orchestration, highlight key ColonyOS features, and showcase example applications in managing quantum computations. ColonyOS has the potential to streamline hybrid calculations and accelerate the path towards quantum-accelerated supercomputing.
## The need for distributed quantum computing

Quantum computing scientists and engineers across different disciplines need to leverage classical and quantum resources in their workflows. They need to develop and test algorithms locally on their personal computers before scaling them up to more complex environments. This iterative process involves expanding the algorithm in terms of parameters, noise models, system complexity, and other details. The next step is to run the workflow on a quantum computer with a few qubits. Of course, due to the limited accessibility to quantum hardware or the limited qubits available, a common target of that workflow is to run it on a powerful computer, which can be an HPC resource capable of simulating hundreds of qubits with the help of some packages and software development kits. Such solutions have the potential to use extensive resources, making it suitable to distribute the workflow over multiple compute resources. For reusability, users require the ability to access those resources seamlessly, saving time by storing profiles of the hardware that this specific workflow can run on across different European compute infrastructures.

## Leveraging ColonyOS for distributed quantum computing

[ColonyOS](https://colonyos.io/) is an open-source meta-operating system designed to streamline workload execution across diverse and distributed computing environments, including cloud, edge, HPC, and IoT. More details are provided in the [arXiv](https://ar5iv.labs.arxiv.org/html/2403.16486) article. This capability makes it well-suited for managing complex, resource-intensive quantum computing tasks. The software is available under the MIT Licence and can be accessed via [GitHub](https://github.com/colonyos). Comprehensive [tutorial notebooks](https://github.com/colonyos/tutorials) are also available to facilitate onboarding.

## Key features of ColonyOS that would help drive quantum-accelerated supercomputing

### Distributed microservice architectures

ColonyOS employs a microservices architecture, where independent executors handle specific tasks. This design supports distributed quantum computing by allowing quantum tasks to be executed across geographically dispersed quantum and classical computing resources in a hybrid fashion. Executors can be deployed independently and scaled horizontally, ensuring efficient parallel processing.

### Workflow orchestration

The platform enables users to define complex, multi-step workflows across distributed executors. This is particularly beneficial for quantum computing applications, which often require iterative execution of quantum circuits, optimisation steps (e.g., variational quantum eigensolver (VQE) algorithms), and hybrid quantum-classical computations. ColonyOS manages dependencies and execution sequencing, ensuring seamless operation across diverse computational systems.

### Scalability

Given the potential for node failures in distributed infrastructures, ColonyOS is designed to reassign tasks dynamically if an executor fails. This approach minimises computation disruptions and enhances overall system reliability.

### Platform-Agnostic Integration

ColonyOS can operate across multiple platforms, including cloud services and HPC environments. This flexibility aligns with the hybrid quantum-classical infrastructures often required for quantum computing workflows, allowing for efficient orchestration of tasks on both classical supercomputers and quantum processors.

The distributed architecture, task orchestration capabilities, and scalability of ColonyOS make it a powerful solution for managing quantum computing workflows. By leveraging ColonyOS, users can efficiently coordinate tasks across quantum and classical computing environments, accelerating the development and deployment of quantum algorithms that are advancing toward further use in quantum-accelerated supercomputing.


## Example: orchestrating and visualising quantum computing workflows with ColonyOS
---

The following snippets are related to an example where ColonyOS orchestrated Qiskit variational calculations with different noise models. ColonyOS generated multiple simulations accounting for different noise models. For more implementation details and related code, please visit the blog post [here](https://www.ekprojectjournal.com/doku.php?id=projects:quantum:distributed).

ColonyOS serialises Qiskit objects, metrics, and metadata from each part of the workflow into an SQLite database. This database is then exposed to localhost via a simple Flask API, which connects to a React frontend that presents two key views of the results data. Both views display the same data and allow ranking across a set of metrics but do so in different ways:

The first way is through the metrics table—a simple (in-development) table that displays each noise simulation computation along with data from its related variational simulation.

<!-- ![Metrics table](img/metrics_table.png)
*Figure 1: The ColonyOS metrics table displaying results from variational simulations with different noise models.* -->
<figure>
  <img src="img/metrics_table.png" alt="Metrics table showing noise model results">
  <figcaption><em>Figure 1: The ColonyOS metrics table displaying results from variational simulations with different noise models.</em></figcaption>
</figure>

The second way is through a workflow graph showing how each step in the workflow is connected and which steps depend on its information.

<!-- ![Workflow unfiltered graph](img/graph_unfiltered.png)
*Figure 2: The ColonyOS workflow graph visualising the connections between different computation steps.* -->
<figure>
  <img src="img/graph_unfiltered.png" alt="Unfiltered workflow graph showing node connections">
  <figcaption><em>Figure 2: The ColonyOS workflow graph visualising the connections between different computation steps.</em></figcaption>
</figure>

Here, the legend explains which part of the calculation workflow the nodes correspond to. A node information panel displays metrics of the selected node. It allows one to compute rankings across nodes (similar to the metrics table) while rescaling and labelling nodes as a function of rank, as seen here:

<!-- ![Workflow filtered graph](img/graph_filtered.png)
*Figure 3: The workflow graph ranked based on specific metrics, highlighting performance differences.* -->
<figure>
  <img src="img/graph_filtered.png" alt="Filtered workflow graph showing ranked nodes">
  <figcaption><em>Figure 3: The workflow graph ranked based on specific metrics, highlighting performance differences.</em></figcaption>
</figure>

With more complicated systems and calculations, the database could present a denser graph providing easily searchable sets of data.

---

These examples highlight ongoing efforts to integrate [ColonyOS](https://colonyos.io/) into quantum computing workflows—a promising step toward distributed computing orchestration. This work also leveraged graph data analytics to view and analyse quantum computation outcomes. [ColonyOS](https://colonyos.io/), as part of [the European Compute Continuum Initiative](https://eucloudedgeiot.eu/decentralised-edge-to-cloud-computing-with-colonyos-recording-now-available/), could become a vital part at the orchestration layer for the [EuroHPC-JU](https://eurohpc-ju.europa.eu/index_en) hybrid quantum-classical infrastructure, enabling seamless utilisation of resources like [LUMI-Q](https://eurohpc-ju.europa.eu/advancing-european-quantum-computing-signature-procurement-contract-eurohpc-quantum-computer-located-2024-09-26_en) and [MareNostrum Q](https://eurohpc-ju.europa.eu/signature-procurement-contract-eurohpc-quantum-computer-located-spain-2025-01-28_en) with the existing HPC classical resources. To explore ColonyOS further, check out the [GitHub repository](https://github.com/colonyos) and the [available tutorials](https://github.com/colonyos/tutorials).

---

## Acknowledgement

This blog post is based on work by [Erik Källman](https://www.ri.se/sv/person/erik-kallman), first presented at the [Nordic Quantum Autumn School 2024](https://enccs.github.io/qas2024/_downloads/e7a4c465a0e6318304e776636c9d317f/QAS-COS.pdf) and further elaborated in an accompanying blog post [here](https://www.ekprojectjournal.com/doku.php?id=projects:quantum:distributed). The original content can be found in [Erik's blog](https://www.ekprojectjournal.com/doku.php?id=projects:quantum:distributed). The concepts and implementations have been adapted and expanded with permission to showcase the potential of [ColonyOS](https://colonyos.io/) in distributed quantum computing workflows. We thank [Erik Källman](https://www.ri.se/sv/person/erik-kallman) for his work in this area and for sharing his insights during the [Nordic Quantum Autumn School 2024](https://enccs.github.io/qas2024/cos/).

---