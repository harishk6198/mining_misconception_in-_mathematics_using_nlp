import trafilatura


def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    # Send a request to the website
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    return text

# Create a simple Flask route to use this functionality
def create_scraper_route(app):
    """
    Adds a scraper route to the given Flask app
    """
    @app.route('/scrape', methods=['GET', 'POST'])
    def scrape():
        from flask import request, render_template, jsonify
        
        if request.method == 'POST':
            url = request.form.get('url')
            if url:
                try:
                    content = get_website_text_content(url)
                    return jsonify({
                        'status': 'success',
                        'content': content if content else 'No content extracted from the URL.'
                    })
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Error: {str(e)}'
                    })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No URL provided.'
                })
                
        return render_template('scrape.html')