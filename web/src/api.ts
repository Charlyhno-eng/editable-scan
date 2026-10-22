export async function errorMessage(response: Response) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : 'Some changes are invalid. Check the text and its position.';
  } catch {
    return 'The server is unavailable. Try again shortly.';
  }
}
