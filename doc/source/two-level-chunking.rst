Two level chunking
==================

Why is it important?
--------------------

When implementing data containers supporting transparent compression, a common technique is to split the containers in so-called chunks.  This allows for avoiding to decompress the whole container for accessing just a subset of elements.  Here it is an example of such a chunked data container for an array of shape `(300, 400, 400)` and whose chunk shape is `(100, 100, 100)`:

.. image:: images/one-level-chunking.png
  :alt: One level chunking

Still, one of the drawbacks of chunking is that it stills requires to retrieve complete chunks even when we are requiring a thinner slice of data.  For example, if we require the plane of data where `dim0 == 140`, we should be reading (and decompressing) the next (rather big) slice:

.. image:: images/one-level-chunking-slice.png
  :alt: Slice for one level chunking

With a two level chunking, we are splitting the original chunking again, and we are getting another subdivision of chunks:

.. image:: images/two-level-chunking.png
  :alt: Two level chunking

Now the chunks have been split in so-called blocks. In this case, the shape of these blocks is `(25, 25, 25)`, so that means that a single chunk is made by `4 * 4 * 4 = 64` blocks.

With that, let's see the amount of data that needs to be retrieved (and decompressed) for the plane `dim0 == 140`:

.. image:: images/two-level-chunking-slice.png
  :alt: Slice for one and two level chunking

This is essentially why the two-level chunking can lead to less reads and hence, better performance.  See below for some concrete figures based on actual datasets.


Some benchmarks
---------------

