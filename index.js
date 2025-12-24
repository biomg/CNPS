export default {
  async fetch(request) {
    return new Response('Worker is working!', {
      headers: { 'content-type': 'text/plain' }
    });
  }
};
