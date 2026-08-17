Contributing
============

We warmly welcome contributions from the community to improve TMMax. This project thrives thanks to the collaborative effort of researchers, engineers, and enthusiasts in the field of optical multilayer thin-film simulations. Whether you are fixing a bug, suggesting a new feature, improving documentation, or enhancing performance, your input is highly valued and appreciated.

Ways You Can Contribute
-----------------------

1. **Reporting Issues**  
   If you encounter a bug, unexpected behavior, or have a question about the functionality of TMMax, please report it by `opening an issue <https://github.com/bahremsd/tmmax/issues>`_. Providing detailed information, including code snippets, error messages, and steps to reproduce the issue, helps the community address it more efficiently.

2. **Suggesting Features**  
   We are always looking to improve TMMax with new features and optimizations. If you have an idea that could enhance the functionality, usability, or performance of TMMax, please share it via an issue. This opens a collaborative discussion and allows the community to provide feedback, refine the idea, and implement it effectively.

3. **Contributing Code**  
   Contributions of code are highly encouraged. This can include bug fixes, optimizations, new features, or enhancements to existing functionality. To contribute code:

   - Fork the repository on GitHub.
   - Create a new branch with a descriptive name for your work.
   - Implement your changes, ensuring that they follow the coding standards and style conventions of TMMax.
   - Test your changes thoroughly using the provided test suite (see section below).
   - Submit a pull request, providing a clear description of the modifications and their purpose.

   To set up a local development environment:

   .. code-block:: bash

      # 1. Clone the repository
      git clone https://github.com/bahremsd/tmmax.git
      cd tmmax

      # 2. Create and activate a virtual environment (optional but recommended)
      python -m venv .venv
      source .venv/bin/activate     # On Windows use: .venv\\Scripts\\activate

      # 3. Install the project with development and test dependencies
      pip install -e ".[dev,test]"

4. **Improving Documentation**  
   Well-written documentation is crucial for the adoption and usability of TMMax. You can help by:

   - Clarifying explanations.
   - Adding examples or tutorials.
   - Correcting typographical errors.
   - Updating information about new features or changes.

5. **Community Engagement**  
   Participation in discussions, helping others solve problems, and providing feedback on ongoing contributions strengthens the TMMax community. Your engagement ensures that the repository continues to grow as a collaborative and inclusive platform.


Testing and Development
-----------------------

TMMax uses **pytest** for automated testing and `pyproject.toml` for dependency management.  
Before submitting code, please ensure that all tests pass successfully.

Run the test suite using one of the following commands:

.. code-block:: bash

   # Run all tests
   pytest

   # Run tests with detailed output
   pytest -v

   # Run tests and generate a coverage report
   pytest --cov=tmmax tests/

   # Run only a specific test file or function
   pytest tests/test_imports.py::test_import_visualize_material_properties

Each command helps contributors verify that their changes maintain functionality and numerical consistency across different simulation modules.

You can also run project-defined tasks (if configured in ``pyproject.toml``) using Hatch or Tox:

.. code-block:: bash

   # Example using hatch
   hatch run dev       # Run development build
   hatch run test      # Run the test suite

Linting and Code Style
----------------------

Before submitting a pull request, ensure that your code follows the TMMax style conventions and passes linting:

.. code-block:: bash

   # Run static analysis
   ruff check tmmax/

   # Automatically fix minor linting issues
   ruff check tmmax/ --fix

   # Format code using black
   black tmmax/


Guidelines for Contributions
----------------------------

- Ensure your contributions are compatible with Python 3 and follow the TMMax code conventions.
- Include appropriate tests and validation for any new functionality.
- Maintain clear, concise, and professional documentation for your additions or modifications.
- Be respectful and considerate in all discussions and feedback.


Contact and Collaboration
-------------------------

For collaboration inquiries or questions that do not fit into GitHub issues, you can contact the maintainer directly via email. We are eager to work with individuals who are passionate about optical thin-film simulations and high-performance computational methods. Collaborative contributions not only improve TMMax but also foster knowledge sharing and innovation within the scientific community.

We appreciate your interest and efforts in contributing to TMMax. Your involvement helps ensure that the project remains cutting-edge, reliable, and accessible to researchers, engineers, and students worldwide.
