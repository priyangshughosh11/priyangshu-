import { render, screen } from '@testing-library/react';
import App from './App';

test('renders StockAI brand name', () => {
  render(<App />);
  const brandElement = screen.getByText(/StockAI/i);
  expect(brandElement).toBeInTheDocument();
});
