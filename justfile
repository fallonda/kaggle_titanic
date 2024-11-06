default:
    just --list

# Delete targets objects cache
delete_objects:
    rm -r ${PWD}/_targets/objects

# pip freeze requirements
freeze:
    pip freeze > requirements.txt