Documentation Style Guide
=========================

Our project's documentation exclusively utilizes reStructuredText (reST), the native markup language of Sphinx, to ensure a consistent and richly detailed presentation of information. This decision aligns with our commitment to leveraging the full spectrum of features offered by Sphinx, enhancing both the developer and user experience through comprehensive documentation practices.

Why reStructuredText (reST)?
-----------------------------

- **Comprehensive Markup Capabilities**: reST allows for intricate documentation features, including hyperlinks, cross-references, emphasis, and structured lists, providing a versatile toolset for detailed documentation.

- **Integrated Sphinx Support**: As the backbone of Sphinx documentation, reST ensures seamless integration and optimal use of Sphinx's capabilities, such as automatic indexing, syntax highlighting, and extensible directives.

- **Uniformity Across Documentation**: By standardizing on reST, we maintain a consistent style and format across all documentation, facilitating easier updates, contributions, and navigation.

Function Documentation Example
-------------------------------

To illustrate how functions should be documented within our codebase, consider the example of a simple function, `calculate_area`, which calculates the area of a rectangle:

.. code-block:: python

   def calculate_area(length, width):
       """Calculates the area of a rectangle given its length and width."""
       return length * width

Here's how this function should be documented in reStructuredText (reST) format:

.. py:function:: calculate_area(length, width)

   Calculates the area of a rectangle given its length and width.

   :param length: The length of the rectangle, a floating-point number.
   :type length: float
   :param width: The width of the rectangle, a floating-point number.
   :type width: float
   :return: The calculated area of the rectangle.
   :rtype: float

   **Example Usage:**

   .. code-block:: python

      area = calculate_area(10, 20)
      print("The area of the rectangle is:", area)

The source code of a documentation can also be accessed by using the button "View source" in the top right corner of the documentation page.

Guidelines for Documentation
-----------------------------

- **Adherence to reST Syntax**: All project documentation, from high-level overviews to in-depth technical descriptions, should strictly follow reST syntax guidelines to maintain consistency and clarity.

- **Clear and Concise Language**: Documentation should be written in a clear, concise manner, making it accessible to both new users and experienced contributors. Aim for simplicity without sacrificing depth and detail.

- **Structured Documentation Approach**: Organize documentation logically, employing reST's structural elements like sections, subsections, and lists to create a coherent and navigable document.

Contributing to Documentation
-----------------------------

We encourage contributions to our documentation. When contributing, please ensure your submissions adhere to the reST format and reflect our documentation standards. Contributions should aim to enhance clarity, improve coverage, or update existing materials to reflect the latest project developments.

By exclusively using reStructuredText (reST) for our documentation, we aim to create a comprehensive, accessible, and cohesive body of documentation that effectively supports our project's goals and community.


