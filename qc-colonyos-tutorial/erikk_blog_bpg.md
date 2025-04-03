# Distributed Quantum Computing workflows with ColonyOS

Quantum computing is evolving rapidly across the globe and Europe is no exception, with investments powering new research frontiers and integrating technologies. But how do researchers and engineers efficiently orchestrate complex quantum computations across diverse hardware, given that classical and quantum hardware need to work side-by-side to achieve the end goal of the calculation(s)? This post explores how ColonyOS can transform and ease distributed quantum computing workflows.

## The need for distributed quantum computing

Quantum computing scientists and engineers across different disciplines need to leverage classical and quantum resources in their workflows. They need to develop and test algorithms locally on their personal computers before scaling them up to more complex environments. This iterative process involves expanding the algorithm in terms of parameters, noise models, system complexity, and other details. The next step is to run the workflow on a quantum computer with a few qubits. Of course, due to the limited accessibility to quantum hardware or the limited qubits available, a common target of that workflow is to run it on a powerful computer, which can be an HPC resource capable of simulating hundreds of qubits with the help of some packages and software development kits. Such solutions have the potential to use extensive resources, making it suitable to distribute the workflow over multiple compute resources. For reusability, users require the ability to access those resources seamlessly, saving time by storing profiles of the hardware that this specific workflow can run on across different European compute infrastructures.


## Leveraging ColonyOS for distributed quantum computing

[ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) is an open-source meta-operating system designed to streamline workload execution across diverse and distributed computing environments, including cloud, edge, HPC, and IoT. This capability makes it well-suited for managing complex, resource-intensive quantum computing tasks. The software is available under the MIT License and can be accessed via [GitHub](https://github.com/colonyos). Comprehensive [tutorial notebooks](https://github.com/colonyos/tutorials) are also available to facilitate onboarding.

## Key features of ColonyOS that would help drive quantum-accelerated supercomputing

### Distributed microservice architectures

[ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) employs a microservices architecture, where independent executors handle specific tasks. This design supports distributed quantum computing by allowing quantum tasks to be executed across geographically dispersed quantum and classical computing resources in a hybrid fashion. Executors can be deployed independently and scaled horizontally, ensuring efficient parallel processing.

### Workflow orchestration

The platform enables users to define complex, multi-step workflows across distributed executors. This is particularly beneficial for quantum computing applications, which often require iterative execution of quantum circuits, optimization steps (e.g., variational quantum eigensolver (VQE) algorithms), and hybrid quantum-classical computations. [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) manages dependencies and execution sequencing, ensuring seamless operation across diverse computational systems.

### Scalability

Given the potential for node failures in distributed infrastructures, [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) is designed to reassign tasks dynamically if an executor fails. This approach minimizes computation disruptions and enhances overall system reliability.

### Platform-Agnostic Integration

[ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) can operate across multiple platforms, including cloud services and HPC environments. This flexibility aligns with the hybrid quantum-classical infrastructures often required for quantum computing workflows, allowing for efficient orchestration of tasks on both classical supercomputers and quantum processors.

The distributed architecture, task orchestration capabilities, and scalability of [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) make it a powerful solution for managing quantum computing workflows. By leveraging [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486), users can efficiently coordinate tasks across quantum and classical computing environments, accelerating the development and deployment of quantum algorithms that are advancing toward further use in quantum-accelerated supercomputing.

---
## Snippets from ColonyOS in action
---
The following snippets are related to an example where [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) orchestrated Qiskit variational calculations with different noise models. [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) generated multiple simulations accounting for different noise models. For more implementation details and related code, please visit the blog post [here](https://www.ekprojectjournal.com/doku.php?id=projects:quantum:distributed).

[ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) serializes Qiskit objects, metrics, and metadata from each part of the workflow into an SQLite database. This database is then exposed to localhost via a simple Flask API, which connects to a React frontend that presents two key views of the results data. Both views display the same data and allow ranking across a set of metrics but do so in different ways:

The first way is through the metrics table—a simple (in-development) table that displays each noise simulation computation along with data from its related variational simulation.

![Metrics table](img/metrics_table.png)

The second way is through a workflow graph showing how each step in the workflow is connected and which steps depend on its information.

![Workflow unfiltered graph](img/graph_unfiltered.png)

Here, the legend explains which part of the calculation workflow the nodes correspond to. A node information panel displays metrics of the selected node. It allows one to compute rankings across nodes (similar to the metrics table) while rescaling and labeling nodes as a function of rank, as seen here:

![Workflow filtered graph](img/graph_filtered.png)

With more complicated systems and calculations, the database could present a denser graph providing easily searchable sets of data.

---

This post has outlined ongoing efforts to integrate [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486) into quantum computing workflows—a promising step toward distributed computing orchestration. This work also leveraged graph data analytics to view and analyze quantum computation outcomes. [ColonyOS](https://ar5iv.labs.arxiv.org/html/2403.16486), as part of [the European Compute Continuum Initiative](https://eucloudedgeiot.eu/decentralised-edge-to-cloud-computing-with-colonyos-recording-now-available/), could become a vital part at the orchestration layer for the [EuroHPC-JU](https://eurohpc-ju.europa.eu/index_en) hybrid quantum-classical infrastructure, enabling seamless utilization of resources like [LUMI-Q](https://eurohpc-ju.europa.eu/advancing-european-quantum-computing-signature-procurement-contract-eurohpc-quantum-computer-located-2024-09-26_en) and [MareNostrum Q](https://eurohpc-ju.europa.eu/signature-procurement-contract-eurohpc-quantum-computer-located-spain-2025-01-28_en)

## Acknowledgement

This blog post is based on work by [Erik Källman](https://www.ri.se/sv/person/erik-kallman), first presented at the [Nordic Quantum Autumn School 2024](https://enccs.github.io/qas2024/cos/). The original content can be found in [Erik's blog](https://www.ekprojectjournal.com/doku.php?id=projects:quantum:distributed). The concepts and implementations have been adapted and expanded with permission to showcase the potential of ColonyOS in distributed quantum computing workflows. We thank [Erik Källman](https://www.ri.se/sv/person/erik-kallman) for his work in this area and for sharing his insights during the Nordic Quantum Autumn School.