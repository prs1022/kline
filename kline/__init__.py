// Function to send message
function sendMessage(message) {
  // Get the textarea element
  const textarea = document.querySelector('.chat-input');
  
  // Set the value of the textarea directly
  textarea.value = message;
  
  // Trigger input event to update any listeners
  const inputEvent = new Event('input', { bubbles: true });
  textarea.dispatchEvent(inputEvent);
  
  // Small delay before clicking the send button to ensure the text is properly set
  setTimeout(() => {
    // Find and click the send button
    const sendButton = document.querySelector('.bl-button--primary');
    sendButton.click();
    console.log('Sent message:', message);
  }, 100);
}

// Set up interval to send message every 3 seconds
const intervalId = setInterval(() => {
  sendMessage('66666');
}, 3000);

// Function to stop sending messages
window.stopSendingMessages = function() {
  clearInterval(intervalId);
  console.log('Stopped sending messages');
};

console.log('Started sending "66666" every 3 seconds. Run stopSendingMessages() to stop.');
