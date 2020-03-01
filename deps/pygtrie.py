# -*- coding: utf-8 -*-
"""Implementation of a trie data structure.

`Trie data structure <http://en.wikipedia.org/wiki/Trie>`_, also known as radix
or prefix tree, is a tree associating keys to values where all the descendants
of a node have a common prefix (associated with that node).

The trie module contains :class:`pygtrie.Trie`, :class:`pygtrie.CharTrie` and
:class:`pygtrie.StringTrie` classes each implementing a mutable mapping
interface, i.e. :class:`dict` interface.  As such, in most circumstances,
:class:`pygtrie.Trie` could be used as a drop-in replacement for
a :class:`dict`, but the prefix nature of the data structure is trie’s real
strength.

The module also contains :class:`pygtrie.PrefixSet` class which uses a trie to
store a set of prefixes such that a key is contained in the set if it or its
prefix is stored in the set.

Features
--------

- A full mutable mapping implementation.

- Supports iterating over as well as deleting of a branch of a trie
  (i.e. subtrie)

- Supports prefix checking as well as shortest and longest prefix
  look-up.

- Extensible for any kind of user-defined keys.

- A PrefixSet supports “all keys starting with given prefix” logic.

- Can store any value including None.

For a few simple examples see ``example.py`` file.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Michal Nazarewicz <mina86@mina86.com>'
__copyright__ = ('Copyright 2014-2017 Google LLC',
                 'Copyright 2018-2019 Michal Nazarewicz <mina86@mina86.com>')


import copy as _copy
import operator as _operator
try:
    import collections.abc as _abc
except ImportError:  # Python 2 compatibility
    import collections as _abc


class ShortKeyError(KeyError):
    """Raised when given key is a prefix of an existing longer key
    but does not have a value associated with itself."""


class _NoChildren(object):
    """Collection representing lack of any children.

    Also acts as an empty iterable and an empty iterator.  This isn’t the
    cleanest designs but it makes various things more concise and avoids object
    allocations in a few places.

    Don’t create objects of this type directly; instead use _EMPTY singleton.
    """
    __slots__ = ()

    def __bool__(self):
        return False
    __nonzero__ = __bool__
    def __len__(self):
        return 0
    def __iter__(self):
        return self
    iteritems = __iter__
    def __next__(self):
        raise StopIteration()
    next = __next__

    def get(self, _step):
        return None

    def add(self, parent, step):
        node = _Node()
        parent.children = _OneChild(step, node)
        return node

    require = add

    def copy(self, _make_copy, _queue):
        return self

    def __deepcopy__(self, memo):
        return self

    # sorted_items, delete and pick_child are not implemented on purpose since
    # they should never be called on a node with no children.


_EMPTY = _NoChildren()


class _OneChild(object):
    """Children collection representing a single child."""

    __solts__ = ('step', 'node')

    def __init__(self, step, node):
        self.step = step
        self.node = node

    def __bool__(self):
        return True
    __nonzero__ = __bool__
    def __len__(self):
        return 1

    def sorted_items(self):
        return [(self.step, self.node)]

    def iteritems(self):
        return iter(((self.step, self.node),))

    def get(self, step):
        return self.node if step == self.step else None

    def add(self, parent, step):
        node = _Node()
        parent.children = _Children((self.step, self.node), (step, node))
        return node

    def require(self, parent, step):
        return self.node if self.step == step else self.add(parent, step)

    def delete(self, parent, _step):
        parent.children = _EMPTY

    def pick_child(self):
        return (self.step, self.node)

    def copy(self, make_copy, queue):
        cpy = _OneChild(make_copy(self.step), self.node.shallow_copy(make_copy))
        queue.append((cpy.node,))
        return cpy


class _Children(dict):
    """Children collection representing more than one child."""

    __slots__ = ()

    def __init__(self, *items):
        super(_Children, self).__init__(items)

    if hasattr(dict, 'iteritems'):  # Python 2 compatibility
        def sorted_items(self):
            items = self.items()
            items.sort()
            return items
    else:
        def sorted_items(self):
            return sorted(self.items())

        def iteritems(self):
            return iter(self.items())

    def add(self, _parent, step):
        self[step] = node = _Node()
        return node

    def require(self, _parent, step):
        return self.setdefault(step, _Node())

    def delete(self, parent, step):
        del self[step]
        if len(self) == 1:
            parent.children = _OneChild(*self.popitem())

    def pick_child(self):
        return next(self.iteritems())

    def copy(self, make_copy, queue):
        cpy = _Children()
        cpy.update((make_copy(step), node.shallow_copy(make_copy))
                   for step, node in self.items())
        queue.append(cpy.values())
        return cpy


class _Node(object):
    """A single node of a trie.

    Stores value associated with the node and dictionary of children.
    """
    __slots__ = ('children', 'value')

    def __init__(self):
        self.children = _EMPTY
        self.value = _EMPTY

    def iterate(self, path, shallow, iteritems):
        """Yields all the nodes with values associated to them in the trie.

        Args:
            path: Path leading to this node.  Used to construct the key when
                returning value of this node and as a prefix for children.
            shallow: Perform a shallow traversal, i.e. do not yield nodes if
                their prefix has been yielded.
            iteritems: A callable taking ``node.children`` as sole argument and
                returning an iterable of children as ``(step, node)`` pair.  The
                callable would typically call ``iteritems`` or ``sorted_items``
                method on the argument depending on whether sorted output is
                desired.

        Yields:
            ``(path, value)`` tuples.
        """
        # Use iterative function with stack on the heap so we don't hit Python's
        # recursion depth limits.
        node = self
        stack = []
        while True:
            if node.value is not _EMPTY:
                yield path, node.value

            if (not shallow or node.value is _EMPTY) and node.children:
                stack.append(iter(iteritems(node.children)))
                path.append(None)

            while True:
                try:
                    step, node = next(stack[-1])
                    path[-1] = step
                    break
                except StopIteration:
                    stack.pop()
                    path.pop()
                except IndexError:
                    return

    def traverse(self, node_factory, path_conv, path, iteritems):
        """Traverses the node and returns another type of node from factory.

        Args:
            node_factory: Callable to construct return value.
            path_conv: Callable to convert node path to a key.
            path: Current path for this node.
            iteritems: A callable taking ``node.children`` as sole argument and
                returning an iterable of children as ``(step, node)`` pair.  The
                callable would typically call ``iteritems`` or ``sorted_items``
                method on the argument depending on whether sorted output is
                desired.

        Returns:
            An object constructed by calling node_factory(path_conv, path,
            children, value=...), where children are constructed by node_factory
            from the children of this node.  There doesn't need to be 1:1
            correspondence between original nodes in the trie and constructed
            nodes (see make_test_node_and_compress in test.py).
        """
        def children():
            """Recursively traverses all of node's children."""
            for step, node in iteritems(self.children):
                yield node.traverse(node_factory, path_conv, path + [step],
                                    iteritems)

        args = [path_conv, tuple(path), children()]

        if self.value is not _EMPTY:
            args.append(self.value)

        return node_factory(*args)

    def equals(self, other):
        """Returns whether this and other node are recursively equal."""
        # Like iterate, we don't recurse so this works on deep tries.
        a, b = self, other
        stack = []
        while True:
            if a.value != b.value or len(a.children) != len(b.children):
                return False
            elif len(a.children) == 1:
                # We know a.children and b.children are both _OneChild objects
                # but pylint doesn’t recognise that: pylint: disable=no-member
                if a.children.step != b.children.step:
                    return False
                a = a.children.node
                b = b.children.node
                continue
            elif a.children:
                stack.append((a.children.iteritems(), b.children))

            while True:
                try:
                    key, a = next(stack[-1][0])
                    b = stack[-1][1][key]
                    break
                except StopIteration:
                    stack.pop()
                except IndexError:
                    return True
                except KeyError:
                    return False

    __bool__ = __nonzero__ = __hash__ = None

    def shallow_copy(self, make_copy):
        """Returns a copy of the node which shares the children property."""
        cpy = _Node()
        cpy.children = self.children
        cpy.value = make_copy(self.value)
        return cpy

    def copy(self, make_copy):
        """Returns a copy of the node structure."""
        cpy = self.shallow_copy(make_copy)
        queue = [(cpy,)]
        while queue:
            for node in queue.pop():
                node.children = node.children.copy(make_copy, queue)
        return cpy

    def __getstate__(self):
        """Get state used for pickling.

        The state is encoded as a list of simple commands which consist of an
        integer and some command-dependent number of arguments.  The commands
        modify what the current node is by navigating the trie up and down and
        setting node values.  Possible commands are:

        * [n, step0, step1, ..., stepn-1, value], for n >= 0, specifies step
          needed to reach the next current node as well as its new value.  There
          is no way to create a child node without setting its (or its
          descendant's) value.

        * [-n], for -n < 0, specifies to go up n steps in the trie.

        When encoded as a state, the commands are flattened into a single list.

        For example::

            [ 0, 'Root',
              2, 'Foo', 'Bar', 'Root/Foo/Bar Node',
             -1,
              1, 'Baz', 'Root/Foo/Baz Node',
             -2,
              1, 'Qux', 'Root/Qux Node' ]

        Creates the following hierarchy::

            -* value: Root
             +-- Foo --* no value
             |         +-- Bar -- * value: Root/Foo/Bar Node
             |         +-- Baz -- * value: Root/Foo/Baz Node
             +-- Qux -- * value: Root/Qux Node

        Returns:
            A pickable state which can be passed to :func:`_Node.__setstate__`
            to reconstruct the node and its full hierarchy.
        """
        # Like iterate, we don't recurse so pickling works on deep tries.
        state = [] if self.value is _EMPTY else [0]
        last_cmd = 0
        node = self
        stack = []
        while True:
            if node.value is not _EMPTY:
                last_cmd = 0
                state.append(node.value)
            stack.append(node.children.iteritems())

            while True:
                step, node = next(stack[-1], (None, None))
                if node is not None:
                    break

                if last_cmd < 0:
                    state[-1] -= 1
                else:
                    last_cmd = -1
                    state.append(-1)
                stack.pop()
                if not stack:
                    state.pop()  # Final -n command is not necessary
                    return state

            if last_cmd > 0:
                last_cmd += 1
                state[-last_cmd] += 1
            else:
                last_cmd = 1
                state.append(1)
            state.append(step)

    def __setstate__(self, state):
        """Unpickles node.  See :func:`_Node.__getstate__`."""
        self.__init__()
        state = iter(state)
        stack = [self]
        for cmd in state:
            if cmd < 0:
                del stack[cmd:]
            else:
                while cmd > 0:
                    parent = stack[-1]
                    stack.append(parent.children.add(parent, next(state)))
                    cmd -= 1
                stack[-1].value = next(state)


class Trie(_abc.MutableMapping):
    """A trie implementation with dict interface plus some extensions.

    Keys used with the :class:`pygtrie.Trie` class must be iterable which each
    component being a hashable objects.  In other words, for a given key,
    ``dict.fromkeys(key)`` must be valid expression.

    In particular, strings work well as trie keys, however when getting them
    back (for example via :func:`Trie.iterkeys` method), instead of strings,
    tuples of characters are produced.  For that reason,
    :class:`pygtrie.CharTrie` or :class:`pygtrie.StringTrie` classes may be
    preferred when using string keys.
    """

    def __init__(self, *args, **kwargs):
        """Initialises the trie.

        Arguments are interpreted the same way :func:`Trie.update` interprets
        them.
        """
        self._root = _Node()
        self._sorted = False
        self.update(*args, **kwargs)

    @property
    def _iteritems(self):
        """Returns function returning iterable over items of its argument.

        Returns:
            A function which returns an iterable over items in a dictionary
            passed to it as an argument.  If child nodes sorting has been
            enabled (via :func:`Trie.enable_sorting` method), returned function
            will go through the items in sorted order.
        """
        return _operator.methodcaller(
            'sorted_items' if self._sorted else 'iteritems')

    def enable_sorting(self, enable=True):
        """Enables sorting of child nodes when iterating and traversing.

        Normally, child nodes are not sorted when iterating or traversing over
        the trie (just like dict elements are not sorted).  This method allows
        sorting to be enabled (which was the behaviour prior to pygtrie 2.0
        release).

        For Trie class, enabling sorting of children is identical to simply
        sorting the list of items since Trie returns keys as tuples.  However,
        for other implementations such as StringTrie the two may behove subtly
        different.  For example, sorting items might produce::

            root/foo-bar
            root/foo/baz

        even though foo comes before foo-bar.

        Args:
            enable: Whether to enable sorting of child nodes.
        """
        self._sorted = bool(enable)

    def clear(self):
        """Removes all the values from the trie."""
        self._root = _Node()

    def update(self, *args, **kwargs):
        """Updates stored values.  Works like :func:`dict.update`."""
        if len(args) > 1:
            raise ValueError('update() takes at most one positional argument, '
                             '%d given.' % len(args))
        # We have this here instead of just letting MutableMapping.update()
        # handle things because it will iterate over keys and for each key
        # retrieve the value.  With Trie, this may be expensive since the path
        # to the node would have to be walked twice.  Instead, we have our own
        # implementation where iteritems() is used avoiding the unnecessary
        # value look-up.
        if args and isinstance(args[0], Trie):
            for key, value in args[0].items():
                self[key] = value
            args = ()
        super(Trie, self).update(*args, **kwargs)

    def copy(self, __make_copy=lambda x: x):
        """Returns a shallow copy of the object."""
        # pylint: disable=protected-access
        cpy = self.__class__()
        cpy.__dict__ = self.__dict__.copy()
        cpy._root = self._root.copy(__make_copy)
        return cpy

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(lambda x: _copy.deepcopy(x, memo))

    @classmethod
    def fromkeys(cls, keys, value=None):
        """Creates a new trie with given keys set.

        This is roughly equivalent to calling the constructor with a ``(key,
        value) for key in keys`` generator.

        Args:
            keys: An iterable of keys that should be set in the new trie.
            value: Value to associate with given keys.

        Returns:
            A new trie where each key from ``keys`` has been set to the given
            value.
        """
        trie = cls()
        for key in keys:
            trie[key] = value
        return trie

    def _get_node(self, key):
        """Returns node for given key.  Creates it if requested.

        Args:
            key: A key to look for.

        Returns:
            ``(node, trace)`` tuple where ``node`` is the node for given key and
            ``trace`` is a list specifying path to reach the node including all
            the encountered nodes.  Each element of trace is a ``(step, node)``
            tuple where ``step`` is a step from parent node to given node and
            ``node`` is node on the path.  The first element of the path is
            always ``(None, self._root)``.

        Raises:
            KeyError: If there is no node for the key.
        """
        node = self._root
        trace = [(None, node)]
        for step in self.__path_from_key(key):
            # pylint thinks node.children is always _NoChildren and thus that
            # we’re assigning None here; pylint: disable=assignment-from-none
            node = node.children.get(step)
            if node is None:
                raise KeyError(key)
            trace.append((step, node))
        return node, trace

    def _set_node(self, key, value, only_if_missing=False):
        node = self._root
        for step in self.__path_from_key(key):
            node = node.children.require(node, step)
        if node.value is _EMPTY or not only_if_missing:
            node.value = value
        return node

    def _set_node_if_no_prefix(self, key):
        steps = iter(self.__path_from_key(key))
        node = self._root
        try:
            while node.value is _EMPTY:
                node = node.children.require(node, next(steps))
        except StopIteration:
            node.value = True
            node.children = _EMPTY

    def __iter__(self):
        return self.iterkeys()

    # pylint: disable=arguments-differ

    def iteritems(self, prefix=_EMPTY, shallow=False):
        node, _ = self._get_node(prefix)
        for path, value in node.iterate(list(self.__path_from_key(prefix)),
                                        shallow, self._iteritems):
            yield (self._key_from_path(path), value)

    def iterkeys(self, prefix=_EMPTY, shallow=False):
        for key, _ in self.iteritems(prefix=prefix, shallow=shallow):
            yield key

    def itervalues(self, prefix=_EMPTY, shallow=False):
        node, _ = self._get_node(prefix)
        for _, value in node.iterate(list(self.__path_from_key(prefix)),
                                     shallow, self._iteritems):
            yield value

    def items(self, prefix=_EMPTY, shallow=False):
        return list(self.iteritems(prefix=prefix, shallow=shallow))

    def keys(self, prefix=_EMPTY, shallow=False):
        return list(self.iterkeys(prefix=prefix, shallow=shallow))

    def values(self, prefix=_EMPTY, shallow=False):
        return list(self.itervalues(prefix=prefix, shallow=shallow))

    def __len__(self):
        return sum(1 for _ in self.itervalues())

    def __bool__(self):
        return self._root.value is not _EMPTY or bool(self._root.children)

    __nonzero__ = __bool__
    __hash__ = None

    HAS_VALUE = 1
    HAS_SUBTRIE = 2

    def has_node(self, key):
        try:
            node, _ = self._get_node(key)
        except KeyError:
            return 0
        return ((self.HAS_VALUE * (node.value is not _EMPTY)) |
                (self.HAS_SUBTRIE * bool(node.children)))

    def has_key(self, key):
        return bool(self.has_node(key) & self.HAS_VALUE)

    def has_subtrie(self, key):
        return bool(self.has_node(key) & self.HAS_SUBTRIE)

    @staticmethod
    def _slice_maybe(key_or_slice):
        if isinstance(key_or_slice, slice):
            if key_or_slice.stop is not None or key_or_slice.step is not None:
                raise TypeError(key_or_slice)
            return key_or_slice.start, True
        return key_or_slice, False

    def __getitem__(self, key_or_slice):
        if self._slice_maybe(key_or_slice)[1]:
            return self.itervalues(key_or_slice.start)
        node, _ = self._get_node(key_or_slice)
        if node.value is _EMPTY:
            raise ShortKeyError(key_or_slice)
        return node.value

    def __setitem__(self, key_or_slice, value):
        key, is_slice = self._slice_maybe(key_or_slice)
        node = self._set_node(key, value)
        if is_slice:
            node.children = _EMPTY

    def setdefault(self, key, value=None):
        return self._set_node(key, value, only_if_missing=True).value

    @staticmethod
    def _cleanup_trace(trace):
        i = len(trace) - 1  # len(path) >= 1 since root is always there
        step, node = trace[i]
        while i and node.value is _EMPTY and not node.children:
            i -= 1
            parent_step, parent = trace[i]
            parent.children.delete(parent, step)
            step, node = parent_step, parent

    def _pop_from_node(self, node, trace):
        value = node.value
        if value is _EMPTY:
            raise ShortKeyError()
        node.value = _EMPTY
        self._cleanup_trace(trace)
        return value

    def pop(self, key, default=_EMPTY):
        try:
            return self._pop_from_node(*self._get_node(key))
        except KeyError:
            if default is not _EMPTY:
                return default
            raise

    def popitem(self):
        if not self:
            raise KeyError()
        node = self._root
        trace = [(None, node)]
        while node.value is _EMPTY:
            # pylint thinks node.children is always _NoChildren which is missing
            # pick_child but we know it must be _OneChild or _Children object:
            step, node = node.children.pick_child()  # pylint: disable=no-member
            trace.append((step, node))
        return (self._key_from_path((step for step, _ in trace[1:])),
                self._pop_from_node(node, trace))

    def __delitem__(self, key_or_slice):
        key, is_slice = self._slice_maybe(key_or_slice)
        node, trace = self._get_node(key)
        if is_slice:
            node.children = _EMPTY
        elif node.value is _EMPTY:
            raise ShortKeyError(key)
        node.value = _EMPTY
        self._cleanup_trace(trace)

    class _NoneStep(object):

        __slots__ = ()

        def __bool__(self):
            return False
        __nonzero__ = __bool__

        def get(self, default=None):
            return default

        is_set = has_subtrie = property(__bool__)
        key = value = property(lambda self: None)

        def __getitem__(self, index):
            if index == 0:
                return self.key
            elif index == 1:
                return self.value
            else:
                raise IndexError('index out of range')

        def __repr__(self):
            return '(None Step)'

    class _Step(_NoneStep):

        __slots__ = ('_trie', '_path', '_pos', '_node', '__key')

        def __init__(self, trie, path, pos, node):
            self._trie = trie
            self._path = path
            self._pos = pos
            self._node = node

        def __bool__(self):
            return True
        __nonzero__ = __bool__

        @property
        def is_set(self):
            return self._node.value is not _EMPTY

        @property
        def has_subtrie(self):
            return bool(self._node.children)

        def get(self, default=None):
            v = self._node.value
            return default if v is _EMPTY else v

        def set(self, value):
            self._node.value = value

        def setdefault(self, value):
            if self._node.value is _EMPTY:
                self._node.value = value
            return self._node.value

        def __repr__(self):
            return '(%r: %r)' % (self.key, self.value)

        @property
        def key(self):
            if not hasattr(self, '_Step__key'):
                # pylint: disable=protected-access
                self.__key = self._trie._key_from_path(self._path[:self._pos])
            return self.__key

        @property
        def value(self):
            v = self._node.value
            if v is _EMPTY:
                raise ShortKeyError(self.key)
            return v

    _NONE_STEP = _NoneStep()

    def walk_towards(self, key):
        node = self._root
        path = self.__path_from_key(key)
        pos = 0
        while True:
            yield self._Step(self, path, pos, node)
            if pos == len(path):
                break
            # pylint thinks node.children is always _NoChildren and thus that
            # we’re assigning None here; pylint: disable=assignment-from-none
            node = node.children.get(path[pos])
            if node is None:
                raise KeyError(key)
            pos += 1

    def prefixes(self, key):
        try:
            for step in self.walk_towards(key):
                if step.is_set:
                    yield step
        except KeyError:
            pass

    def shortest_prefix(self, key):
        return next(self.prefixes(key), self._NONE_STEP)

    def longest_prefix(self, key):
        ret = self._NONE_STEP
        for ret in self.prefixes(key):
            pass
        return ret

    def __eq__(self, other):
        # pylint: disable=protected-access
        return self is other or self._root.equals(other._root)

    def __ne__(self, other):
        return not self == other

    def _str_items(self, fmt='%s: %s'):
        return ', '.join(fmt % item for item in self.iteritems())

    def __str__(self):
        return '%s(%s)' % (type(self).__name__, self._str_items())

    def __repr__(self):
        return '%s([%s])' % (type(self).__name__, self._str_items('(%r, %r)'))

    def __path_from_key(self, key):
        return () if key is _EMPTY else self._path_from_key(key)

    def _path_from_key(self, key):
        return key

    def _key_from_path(self, path):
        return tuple(path)

    def traverse(self, node_factory, prefix=_EMPTY):
        node, _ = self._get_node(prefix)
        return node.traverse(node_factory, self._key_from_path,
                             list(self.__path_from_key(prefix)),
                             self._iteritems)



class StringTrie(Trie):

    def __init__(self, *args, **kwargs):  # pylint: disable=differing-param-doc
        separator = kwargs.pop('separator', '/')
        if not isinstance(separator, getattr(__builtins__, 'basestring', str)):
            raise TypeError('separator must be a string')
        if not separator:
            raise ValueError('separator can not be empty')
        self._separator = separator
        super(StringTrie, self).__init__(*args, **kwargs)

    @classmethod
    def fromkeys(cls, keys, value=None, separator='/'):  # pylint: disable=arguments-differ
        trie = cls(separator=separator)
        for key in keys:
            trie[key] = value
        return trie

    def __str__(self):
        if not self:
            return '%s(separator=%s)' % (type(self).__name__, self._separator)
        return '%s(%s, separator=%s)' % (
            type(self).__name__, self._str_items(), self._separator)

    def __repr__(self):
        return '%s([%s], separator=%r)' % (
            type(self).__name__, self._str_items('(%r, %r)'), self._separator)

    def _path_from_key(self, key):
        return key.split(self._separator)

    def _key_from_path(self, path):
        return self._separator.join(path)


class PrefixSet(_abc.MutableSet):

    def __init__(self, iterable=(), factory=Trie, **kwargs):
        super(PrefixSet, self).__init__()
        self._trie = factory(**kwargs)
        for key in iterable:
            self.add(key)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        # pylint: disable=protected-access
        cpy = self.__class__()
        cpy.__dict__ = self.__dict__.copy()
        cpy._trie = self._trie.__copy__()
        return cpy

    def __deepcopy__(self, memo):
        # pylint: disable=protected-access
        cpy = self.__class__()
        cpy.__dict__ = self.__dict__.copy()
        cpy._trie = self._trie.__deepcopy__(memo)
        return cpy

    def clear(self):
        self._trie.clear()

    def __contains__(self, key):
        return bool(self._trie.shortest_prefix(key)[1])

    def __iter__(self):
        return self._trie.iterkeys()

    def iter(self, prefix=_EMPTY):
        if prefix is _EMPTY:
            return iter(self)
        elif self._trie.has_node(prefix):
            return self._trie.iterkeys(prefix=prefix)
        elif prefix in self:
            # Make sure the type of returned keys is consistent.
            # pylint: disable=protected-access
            return (
                self._trie._key_from_path(self._trie._path_from_key(prefix)),)
        else:
            return ()

    def __len__(self):
        return len(self._trie)

    def add(self, value):
        # We're friends with Trie;  pylint: disable=protected-access
        self._trie._set_node_if_no_prefix(value)

