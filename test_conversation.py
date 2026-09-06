from src.nl2sql.conversation import conversation_manager

sid = "test-session-1"

print("Turn 1 context (should be empty):", repr(conversation_manager.get_context_string(sid)))

conversation_manager.add_turn(
    sid,
    question="What is the average credit amount for approved loans?",
    sql="SELECT AVG(AMT_CREDIT) FROM application_train WHERE TARGET = 0",
    answer="The average credit amount for approved loans is 599,026 RUB."
)

print("\nTurn 2 context:")
print(conversation_manager.get_context_string(sid))

conversation_manager.add_turn(
    sid,
    question="Now break that down by gender",
    sql="SELECT CODE_GENDER, AVG(AMT_CREDIT) FROM application_train WHERE TARGET = 0 GROUP BY CODE_GENDER",
    answer="For approved loans, average credit amount is 598,344 RUB for females and 605,111 RUB for males."
)

print("\nTurn 3 context:")
print(conversation_manager.get_context_string(sid))

print("\nSession exists check:", conversation_manager.session_exists(sid))
print("Unknown session exists check:", conversation_manager.session_exists("nonexistent-session"))

conversation_manager.clear_session(sid)
print("\nAfter clear_session, context (should be empty):", repr(conversation_manager.get_context_string(sid)))
print("After clear_session, session_exists (should be False):", conversation_manager.session_exists(sid))

sid2 = "test-session-2"
for i in range(1, 6):
    conversation_manager.add_turn(
        sid2,
        question=f"Question {i}",
        sql=f"SELECT {i}",
        answer=f"Answer {i}"
    )

print("\nAfter adding 5 turns (max_turns=3), context should only show last 3:")
print(conversation_manager.get_context_string(sid2))