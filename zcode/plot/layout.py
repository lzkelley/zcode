"""
"""
import warnings

import matplotlib as mpl
import numpy as np
import six

from . plot_const import _PAD


__all__ = ["backdrop", "extent", "full_extent", "position_to_extent", "rect_for_inset",
           "transform"]


def backdrop(fig, obj, pad=0.0, union=False, group=False, draw=True, **kwargs):
    """Draw a rectangle behind the full extent of the given object.

    Example
    -------
    >>> zplot.backdrop(fig, ax, pad=0.02, union=True,
                       facecolor='white', edgecolor='none', zorder=5, alpha=0.8)
    """
    if union:
        bboxes = [full_extent(obj, pad=pad, invert=fig.transFigure)]
    else:
        bboxes = _extents(obj, pad=pad, invert=fig.transFigure, group=group)

    pats = []
    for bbox in bboxes:
        rect = mpl.patches.Rectangle([bbox.xmin, bbox.ymin], bbox.width, bbox.height,
                                     transform=fig.transFigure, **kwargs)
        if draw: fig.patches.append(rect)
        pats.append(rect)

    if len(pats) == 1: return pats[0]
    return pats


def extent(ax, pad=0.0, invert=None, fig=None):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    From: 'stackoverflow.com/questions/14712665/'
    """
    # Draw text objects so extents are defined
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent().expanded(1.0 + pad, 1.0 + pad)
    if invert:
        bbox = bbox.transformed(invert.inverted())

    return bbox


def full_extent(ax, pad=0.0, invert=None):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    From: 'stackoverflow.com/questions/14712665/'
    """
    # Draw text objects so extents are defined
    ax.figure.canvas.draw()
    if isinstance(ax, mpl.axes.Axes):
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # items += [ax, ax.title]
        use_items = []
        for item in items:
            if isinstance(item, mpl.text.Text) and len(item.get_text()) == 0: continue
            use_items.append(item)
        # bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in use_items])
        bbox = _set_extents(use_items, union=True, pad=0.0)
    elif isinstance(ax, mpl.legend.Legend):
        bbox = ax.get_frame().get_bbox()
    else:
        err_str = "Unrecognized type of `ax` = '{}'.  Currently support axes and legends.".format(
            type(ax))
        raise ValueError(err_str)

    bbox = bbox.expanded(1.0 + pad, 1.0 + pad)
    if invert:
        bbox = bbox.transformed(invert.inverted())

    return bbox


def position_to_extent(fig, ref, loc, item=None, pad=0.0, halign='left', valign='lower'):
    """Reposition axis so that origin of 'full_extent' is at given `loc`.
    """
    if item is None:
        item = ref

    bbox = full_extent(ref, pad=pad, invert=fig.transFigure)
    # try:
    ax_bbox = ref.get_position()
    # except AttributeError:
    #     ax_bbox = ref.get_bbox_to_anchor()
    #     print("old_loc = ", ax_bbox)

    if halign.startswith('r'):
        dx = ax_bbox.x1 - bbox.x1
    elif halign.startswith('l'):
        dx = ax_bbox.x0 - bbox.x0
    elif halign.startswith('c'):
        dx = 0.5*(ax_bbox.x0 + ax_bbox.x1) - 0.5*(bbox.x0 + bbox.x1)
    else:
        raise ValueError("`halign` = '{}' must start with 'l' or 'r'.".format(halign))

    if valign.startswith('l'):
        dy = ax_bbox.y0 - bbox.y0
    elif valign.startswith('u'):
        dy = ax_bbox.y1 - bbox.y1
    elif valign.startswith('c'):
        dy = 0.5*(ax_bbox.y0 + ax_bbox.y1) - 0.5*(bbox.y0 + bbox.y1)
    else:
        raise ValueError("`valign` = '{}' must start with 'l' or 'u'.".format(valign))

    if len(loc) == 2:
        new_loc = [loc[0]+dx, loc[1]+dy, ax_bbox.width, ax_bbox.height]
    elif len(loc) == 4:
        new_loc = [loc[0]+dx, loc[1]+dy, loc[2], loc[3]]
    else:
        raise ValueError("`loc` must be 2 or 4 long: [x, y, (width, height)]")

    # try:
    item.set_position(new_loc)
    # except AttributeError:
    #     print("new_loc = ", new_loc)
    #     item.set_bbox_to_anchor(new_loc)

    return


def rect_for_inset(parent, loc='tl', width=None, height=None,
                   width_frac=0.25, height_frac=0.25, pad=None):
    """Construct the rectangle to describe an inset axes relative to the parent object.
    """
    pos = parent.get_position()
    wid = pos.width * width_frac if (width is None) else width
    hit = pos.height * height_frac if (height is None) else height

    if pad is None:
        pad = _PAD

    if loc[0] == 'b':
        yy = pos.y0 + pad
    elif loc[0] == 't':
        yy = pos.y1 - pad - hit
    else:
        raise ValueError("`loc[0]` must be ['b', 't']!")

    if loc[1] == 'l':
        xx = pos.x0 + pad
    elif loc[1] == 'r':
        xx = pos.x1 - pad - wid
    else:
        raise ValueError("`loc[1]` must be ['l', 'r']!")

    rect = [xx, yy, wid, hit]
    return rect


def transform(ax, trans, fig=None):
    """Create a (blended) transformation.
    """
    if np.size(trans) == 1 and isinstance(trans, six.string_types):
        trans = [trans]
    elif np.size(trans) != 2:
        raise ValueError("`trans` = '{}' must be a string or pair of strings.".format(trans))

    forms = []
    for tt in trans:
        if tt.startswith('ax'):
            forms.append(ax.transAxes)
        elif tt.startswith('data'):
            forms.append(ax.transData)
        elif tt.startswith('fig'):
            if fig is None:
                raise ValueError("trans = '{}' requires a `fig` object".format(tt))
            forms.append(fig.transFigure)
        else:
            raise ValueError("Unrecognized transform '{}'".format(tt))

    if len(forms) == 1:
        transform = forms[0]
    else:
        transform = mpl.transforms.blended_transform_factory(forms[0], forms[1])

    return transform


def _loc_str_to_pars(loc, x=None, y=None, halign=None, valign=None, pad=_PAD):
    """Convert from a string location specification to the specifying parameters.

    If any of the specifying parameters: {x, y, halign, valign}, are 'None', they are set to
    default values.

    Returns
    -------
    x : float
    y : float
    halign : str
    valign : str

    """
    _valid_loc = [['t', 'u', 'b', 'l', 'c'], ['l', 'r', 'c']]
    for ii, (ll, vv) in enumerate(zip(loc, _valid_loc)):
        if ll not in vv:
            err = "Unrecognized `loc`[{}] = '{}' (`loc` = '{}').".format(ii, ll, loc)
            err += "\n\t`loc`[{}] must be one of '{}'".format(ii, vv)
            raise ValueError(err)

    pad = np.atleast_1d(pad)
    if pad.size == 1:
        pad = np.concatenate([pad, pad])

    if loc[0] == 't' or loc[0] == 'u':
        if valign is None:
            valign = 'top'
        if y is None:
            y = 1 - pad[1]
    elif loc[0] == 'b' or loc[0] == 'l':
        if valign is None:
            valign = 'bottom'
        if y is None:
            y = pad[1]
    elif loc[0] == 'c':
        if valign is None:
            valign = 'center'
        if y is None:
            y = 0.5

    if loc[1] == 'l':
        if halign is None:
            halign = 'left'
        if x is None:
            x = pad[0]
    elif loc[1] == 'r':
        if halign is None:
            halign = 'right'
        if x is None:
            x = 1 - pad[0]
    elif loc[1] == 'c':
        if halign is None:
            halign = 'center'
        if x is None:
            x = 0.5

    return x, y, halign, valign


def _extents(ax, pad=0.0, invert=None, group=False):
    """Get the extents of an axes, including axes labels, tick labels, and titles.

    Adapted From: 'stackoverflow.com/questions/14712665/'
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    if group:
        items = [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        bboxes = [_set_extents(ax.get_xticklabels(), pad=pad, union=True)]
        bboxes += [_set_extents(ax.get_yticklabels(), pad=pad, union=True)]
        bboxes += _set_extents(items, pad=pad, union=False)
    else:
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        # bboxes = [it.get_window_extent().expanded(1.0 + pad, 1.0 + pad) for it in items]
        bboxes = _set_extents(items, pad=pad, union=False)

    if invert:
        bboxes = [bb.transformed(invert.inverted()) for bb in bboxes]

    return bboxes


def _parse_align(halign=None, valign=None):
    if halign is None:
        halign = 'center'
    if valign is None:
        valign = 'top'

    if halign.startswith('l'):
        halign = 'left'
    elif halign.startswith('c'):
        halign = 'center'
    elif halign.startswith('r'):
        halign = 'right'

    if valign.startswith('t'):
        valign = 'top'
    elif valign.startswith('c'):
        valign = 'center'
    elif valign.startswith('b'):
        valign = 'bottom'

    if valign == 'upper':
        warnings.warn("Use `'top'` not `'upper'`!")
        valign = 'top'

    if valign == 'lower':
        warnings.warn("Use `'bottom'` not `'lower'`!")
        valign = 'bottom'
    return halign, valign


def _set_extents(items, pad=0.0, union=False):
    bboxes = [it.get_window_extent().expanded(1.0 + pad, 1.0 + pad) for it in items]
    if union:
        bboxes = mpl.transforms.Bbox.union(bboxes)
    return bboxes
