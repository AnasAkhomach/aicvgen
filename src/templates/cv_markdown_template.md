# {{ cv.metadata.name or 'Your Name' }}

{% if cv.metadata.email or cv.metadata.phone or cv.metadata.linkedin %}
**Contact:**
{% if cv.metadata.email %}{{ cv.metadata.email }}{% endif %}{% if cv.metadata.phone %} | {{ cv.metadata.phone }}{% endif %}{% if cv.metadata.linkedin %} | [LinkedIn]({{ cv.metadata.linkedin }}){% endif %}
{% endif %}

{% for section in cv.sections %}
## {{ section.name }}
{% if section.items %}
{% if section.name == 'Key Qualifications' %}
**Skills:** {% for item in section.items %}{{ item.content }}{% if not loop.last %}, {% endif %}{% endfor %}
{% else %}
- {% for item in section.items %}{{ item.content }}
- {% endfor %}
{% endif %}
{% endif %}
{% if section.subsections %}
{% for sub in section.subsections %}
### {{ sub.name }}
{% if sub.metadata and sub.metadata.company %}*{{ sub.metadata.company }}*{% endif %}{% if sub.metadata and sub.metadata.duration %} ({{ sub.metadata.duration }}){% endif %}
{% for item in sub.items %}- {{ item.content }}
{% endfor %}
{% endfor %}
{% endif %}
{% endfor %}
