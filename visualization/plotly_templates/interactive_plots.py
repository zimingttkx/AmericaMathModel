"""Plotly 交互式图表模板 - 数学建模常用"""
import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("请安装 plotly: pip install plotly")
    raise


def interactive_scatter(df, x, y, color=None, size=None, title='Scatter Plot'):
    """交互式散点图"""
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title,
                     hover_data=df.columns)
    fig.update_layout(template='plotly_white')
    return fig


def interactive_line(df, x, y, color=None, title='Line Chart'):
    """交互式折线图"""
    fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)
    fig.update_layout(template='plotly_white')
    return fig


def interactive_bar(df, x, y, color=None, title='Bar Chart', barmode='group'):
    """交互式柱状图"""
    fig = px.bar(df, x=x, y=y, color=color, title=title, barmode=barmode)
    fig.update_layout(template='plotly_white')
    return fig


def interactive_heatmap(df, title='Heatmap'):
    """交互式热力图"""
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto='.2f', title=title, color_continuous_scale='RdBu_r')
    fig.update_layout(template='plotly_white')
    return fig


def interactive_3d_scatter(df, x, y, z, color=None, title='3D Scatter'):
    """3D 散点图"""
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title)
    fig.update_layout(template='plotly_white')
    return fig


def interactive_surface(X, Y, Z, title='3D Surface'):
    """3D 曲面图"""
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(title=title, template='plotly_white',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    return fig


def interactive_histogram(df, x, nbins=30, title='Histogram'):
    """交互式直方图"""
    fig = px.histogram(df, x=x, nbins=nbins, title=title, marginal='box')
    fig.update_layout(template='plotly_white')
    return fig


def interactive_box(df, x, y, title='Box Plot'):
    """交互式箱线图"""
    fig = px.box(df, x=x, y=y, title=title, points='outliers')
    fig.update_layout(template='plotly_white')
    return fig


def multi_panel_figure(data_list, titles, rows=2, cols=2):
    """多面板图"""
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for i, (data, trace_type) in enumerate(data_list):
        row = i // cols + 1
        col = i % cols + 1
        if trace_type == 'scatter':
            fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='markers'), row=row, col=col)
        elif trace_type == 'line':
            fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='lines'), row=row, col=col)
        elif trace_type == 'bar':
            fig.add_trace(go.Bar(x=data['x'], y=data['y']), row=row, col=col)
    fig.update_layout(template='plotly_white', showlegend=False)
    return fig


def animated_scatter(df, x, y, animation_frame, color=None, title='Animated Scatter'):
    """动画散点图"""
    fig = px.scatter(df, x=x, y=y, animation_frame=animation_frame, color=color,
                     title=title, range_x=[df[x].min(), df[x].max()],
                     range_y=[df[y].min(), df[y].max()])
    fig.update_layout(template='plotly_white')
    return fig


def radar_chart(categories, values, title='Radar Chart'):
    """雷达图"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                   fill='toself', name='Values'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=title,
                      template='plotly_white')
    return fig


if __name__ == '__main__':
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    fig = interactive_scatter(df, 'x', 'y', color='category', title='Demo Scatter')
    fig.write_html('plotly_demo.html')
    print("Demo saved to plotly_demo.html")
