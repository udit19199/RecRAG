Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 1 items.

=== ITEM id=805b5fdcc898390b ===
Gold sentences and facts:
Sentence 0: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 0.0: Al Saadi said her frustration with Meta is common among Palestinians.
  Fact 0.1: Al Saadi said the frustration with Meta is common among Palestinians in occupied territories.
  Fact 0.2: Al Saadi said the frustration with Meta is common among Palestinians across the diaspora.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores.
  Fact 1.1: Meta wants to shift the burden of monitoring social media usage among teens to parents.

Retrieved items:
<item 1>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“It feeds into the bias against Palestinian narrative genuinely. It really takes the balance against Palestinians in a situation where there’s a huge asymmetry
of power.”

A history of suppression

Content about Palestine is disproportionately scrutinized, as demonstrated during the last severe violent outbreak between Hamas and Israel two years ago. Amid
the violence following the May 2021 court ruling to evict Palestinian families from Sheikh Jarrah, a neighborhood in occupied East Jerusalem, users across
Facebook and Instagram accused Meta of taking down posts and suspending accounts that voiced support for Palestinians.

The digital rights nonprofit Electronic Frontier Foundation (EFF) described Meta’s actions in 2021 as “systemic censorship of Palestinian voices.” In its 2022
report of Palestinian digital rights, 7amleh said that Meta is “still the most restricting company” compared to other social media giants in the extent of its
moderation of the Palestinian digital space.

Graph facts:
- How Chinese firm linked to repression of Uyghurs aids Israeli surveillance in West Bank -MENTIONS-> occupied East Jerusalem
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Electronic Frontier Foundation
- Jalal Abukhater -WORKS_FOR-> 7amleh
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Sheikh Jarrah
- John Perry Barlow -FOUNDED-> Electronic Frontier Foundation

Entities: Palestinian voices, 7amleh, moderation bias, Severe violent outbreak between Hamas and Israel, Sheikh Jarrah, occupied East Jerusalem, Electronic
Frontier Foundation, Palestinian digital rights, Systemic censorship of Palestinian voices, Moderation of the Palestinian digital space, Meta has a moderation
bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices, Meta, TechCrunch, Court ruling to evict Palestinian families from Sheikh Jarrah, Hamas
</item 1>

<item 2>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”

Graph facts:
- Jalal Abukhater -WORKS_FOR-> 7amleh
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> censorship
- Leen Al Saadi -SAID-> censorship
- The people who ruined the internet -MENTIONS-> censorship
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Stone

Entities: censorship, Palestinian digital rights violations, 7amleh, moderation bias, Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing
Palestinian voices, Meta, ceasefire between Hamas and Israel, Colombia, technical issues, Al-Aqsa Mosque, Israeli police, clash between Israeli police and
Ramadan worshippers, Ramadan worshippers, WhatsApp, Facebook employees
</item 2>

<item 3>
Graph query row:
article: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices
passage: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices Instagram users suspect Meta of shadowbanning content about
Palestine. The platform has a history of inherent bias.

Earlier this year, Palestinian-American filmmaker Khitam Jabr posted a handful of Reels about her family’s trip to the West Bank. In the short travel vlogs,
Jabr shared snippets of Palestinian culture, from eating decadent meals to dancing at her niece’s wedding.

“I hadn’t been in a decade, so it’s just like, life abroad,” Jabr told TechCrunch.

“I would get [anti-Palestine] comments,” she recalled. “And I couldn’t respond [to them] or use my account for 24 hours. I wasn’t even posting anything about
the occupation. But fast forward to now and the same shit’s happening.”

In the aftermath of Hamas’ attack on Israelis, Israel’s retaliatory airstrikes and total blockade — cutting access to electricity, water and vital supplies —
have devastated Gaza.
mentioned_entities: ['Palestinian voices', 'Palestinian street art', 'suppression of Palestinian voices', 'Meta’s updated newsroom post', 'Meta identified and
fixed a bug that prevented users from going live', 'Palestinian digital rights violations', 'Palestinian flag emoji', 'Palestinian flag', 'Palestinian
narrative', 'Palestinian digital rights', 'Systemic censorship of Palestinian voices', 'Moderation of the Palestinian digital space', 'Meta', 'Palestinian
Arabic', 'regional Palestinian organizations', 'balance of power between Israelis and Palestinians', 'Meta’s lack of cultural competence', 'Court ruling to
evict Palestinian families from Sheikh Jarrah', 'Palestinians']
entity_connections: [{'from': 'teens’ startup', 'relationship': 'MENTIONS', 'to': 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests
an ad-free tier'}, {'from': 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier', 'relationship': 'MENTIONS', 'to':
'teens’ startup'}, {'from': "Alexandra's parents", 'relationship': 'MENTIONS', 'to': 'At 37, Alexandra found herself broke, heartbroken and living with her
parents'}, {'from': 'parenthood', 'relationship': 'MENTIONS', 'to': 'At 37, Alexandra found herself broke, heartbroken and living with her parents'}, {'from':
'At 37, Alexandra found herself broke, heartbroken and living with her parents', 'relationship': 'MENTIONS', 'to': "Alexandra's parents"}, {'from': 'At 37,
Alexandra found herself broke, heartbroken and living with her parents', 'relationship': 'MENTIONS', 'to': 'parenthood'}, {'from': 'At 37, Alexandra found
herself broke, heartbroken and living with her parents', 'relationship': 'MENTIONS', 'to': 'Rachel Claire moved back in with her parents'}, {'from': 'Rachel
Claire moved back in with her parents', 'relationship': 'MENTIONS', 'to': 'At 37, Alexandra found herself broke, heartbroken and living with her parents'},
{'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'relationship': 'MENTIONS', 'to': 'app store antitrust issues'}, {'from': 'app
store antitrust issues', 'relationship': 'MENTIONS', 'to': 'Epic Games lawsuit against Google over app store antitrust issues'}, {'from': 'Meta layoffs',
'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'Meta layoff announcement', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'Meta cut
11,000 jobs', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'Meta workforce cuts', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'How
the OpenAI fiasco could bolster Meta and the ‘open AI’ movement', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'How the OpenAI fiasco could bolster Meta
and the ‘open AI’ movement', 'relationship': 'MENTIONS', 'to': 'Facebook'}, {'from': 'Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta has a
moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Palestinian street art', 'relationship': 'MENTIONS', 'to': 'Meta
has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'suppression of Palestinian voices', 'relationship':
'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'suppression of Palestinian voices',
'relationship': 'ACCUSED_OF', 'to': 'Meta'}, {'from': 'Meta’s updated newsroom post', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not
just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Meta identified and fixed a bug that prevented users from going live', 'relationship':
'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Meta identified and fixed a bug that
prevented users from going live', 'relationship': 'SAID', 'to': 'Meta'}, {'from': 'Palestinian digital rights violations', 'relationship': 'MENTIONS', 'to':
'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Palestinian flag emoji', 'relationship': 'MENTIONS',
'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Palestinian flag', 'relationship': 'MENTIONS',
'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Palestinian narrative', 'relationship':
'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Palestinian digital rights',
'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Systemic censorship
of Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'},
{'from': 'Moderation of the Palestinian digital space', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s
suppressing Palestinian voices'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship':
'MENTIONS', 'to': 'Palestinian voices'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship':
'MENTIONS', 'to': 'Palestinian street art'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices',
'relationship': 'MENTIONS', 'to': 'suppression of Palestinian voices'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing
Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta’s updated newsroom post'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s
suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta identified and fixed a bug that prevented users from going live'}, {'from': 'Meta has a
moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Palestinian digital rights violations'},
{'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Palestinian flag
emoji'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Palestinian
flag'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Palestinian
narrative'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to':
'Palestinian digital rights'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship':
'MENTIONS', 'to': 'Systemic censorship of Palestinian voices'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian
voices', 'relationship': 'MENTIONS', 'to': 'Moderation of the Palestinian digital space'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’
that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s
suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Palestinian Arabic'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s
suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'regional Palestinian organizations'}, {'from': 'Meta has a moderation bias problem, not just
a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'balance of power between Israelis and Palestinians'}, {'from': 'Meta has a
moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Meta’s lack of cultural competence'},
{'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices', 'relationship': 'MENTIONS', 'to': 'Court ruling to evict
Palestinian families from Sheikh Jarrah'}, {'from': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices',
'relationship': 'MENTIONS', 'to': 'Palestinians'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'How the OpenAI fiasco could bolster Meta and the ‘open
AI’ movement'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian
voices'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Norway seeks to extend ban on Meta’s consentless tracking ads across the EU'}, {'from': 'Meta',
'relationship': 'MENTIONS', 'to': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'},
{'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'}, {'from': 'Meta',
'relationship': 'MENTIONS', 'to': 'Meta and IBM form an AI Alliance, but to what end?'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Meta turned a blind
eye to kids on its platforms for years, unredacted lawsuit alleges'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Meta says it’s prioritizing
livestreaming checks during Israel-Hamas war'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta,
Ford, GM and more'}, {'from': 'Meta', 'relationship': 'MENTIONS', 'to': 'The Ray-Ban Meta smart glasses actually make the future look cool'}, {'from': 'Meta',
'relationship': 'MENTIONS', 'to': 'Meta to offer ad-free subscription in Europe in bid to keep tracking other users'}, {'from': 'Meta', 'relationship': 'SAID',
'to': 'Meta identified and fixed a bug that prevented users from going live'}, {'from': 'Meta', 'relationship': 'SAID', 'to': 'parental approval for teens’ app
downloads'}, {'from': 'Meta', 'relationship': 'SAID', 'to': 'parental approval of teens’ app downloads'}, {'from': 'Meta', 'relationship': 'PARTICIPATED_IN',
'to': 'Meta layoffs'}, {'from': 'Meta', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta layoff announcement'}, {'from': 'Meta', 'relationship':
'PARTICIPATED_IN', 'to': 'Meta cut 11,000 jobs'}, {'from': 'Meta', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta workforce cuts'}, {'from': 'Meta',
'relationship': 'PARTICIPATED_IN', 'to': 'Oslo court rejection of Meta’s arguments'}, {'from': 'Meta', 'relationship': 'PARTICIPATED_IN', 'to': 'privacy
complaint lodged against Meta'}, {'from': 'Meta', 'relationship': 'PARTICIPATED_IN', 'to': 'lawsuit against Meta over harms to teens and young users'}, {'from':
'Meta', 'relationship': 'ACCUSED_OF', 'to': 'suppression of Palestinian voices'}, {'from': 'Meta', 'relationship': 'ACCUSED_OF', 'to': 'harms to teens and young
users'}, {'from': 'Meta', 'relationship': 'ACCUSED_OF', 'to': 'verifiable parental consent'}, {'from': 'Meta', 'relationship': 'ACCUSED_OF', 'to': "Turning a
blind eye to kids on Meta's platforms"}, {'from': 'Meta', 'relationship': 'ACCUSED_OF', 'to': "kids on Meta's platforms"}, {'from': 'Meta', 'relationship':
'MADE', 'to': 'monthly subscription for an ad-free version of Meta’s social networks'}, {'from': 'Meta', 'relationship': 'MADE', 'to': 'Meta ad-free
subscription'}, {'from': 'Meta', 'relationship': 'MADE', 'to': 'Meta’s ad-free subscription'}, {'from': 'Meta', 'relationship': 'MADE', 'to': "Meta's AI-powered
image generator"}, {'from': 'Meta', 'relationship': 'MADE', 'to': 'Imagine with Meta AI'}, {'from': 'Meta', 'relationship': 'OWNS', 'to': 'Facebook'}, {'from':
'Meta', 'relationship': 'PART_OF', 'to': 'Facebook'}, {'from': 'Palestinian Arabic', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not
just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'regional Palestinian organizations', 'relationship': 'MENTIONS', 'to': 'Meta has a moderation
bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'balance of power between Israelis and Palestinians', 'relationship':
'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Meta’s lack of cultural competence',
'relationship': 'MENTIONS', 'to': 'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Local ban on Meta
tracking and profiling users without consent', 'relationship': 'MENTIONS', 'to': 'Norway seeks to extend ban on Meta’s consentless tracking ads across the EU'},
{'from': 'Oslo court rejection of Meta’s arguments', 'relationship': 'MENTIONS', 'to': 'Norway seeks to extend ban on Meta’s consentless tracking ads across the
EU'}, {'from': 'Oslo court rejection of Meta’s arguments', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'Norway seeks to extend ban on Meta’s
consentless tracking ads across the EU', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Norway seeks to extend ban on Meta’s consentless tracking ads
across the EU', 'relationship': 'MENTIONS', 'to': 'Local ban on Meta tracking and profiling users without consent'}, {'from': 'Norway seeks to extend ban on
Meta’s consentless tracking ads across the EU', 'relationship': 'MENTIONS', 'to': 'Oslo court rejection of Meta’s arguments'}, {'from': 'Norway seeks to extend
ban on Meta’s consentless tracking ads across the EU', 'relationship': 'MENTIONS', 'to': 'DPC final decision rejecting Meta’s contractual necessity claim'},
{'from': 'Norway seeks to extend ban on Meta’s consentless tracking ads across the EU', 'relationship': 'MENTIONS', 'to': 'ruling on Meta’s legitimate interests
claim'}, {'from': 'DPC final decision rejecting Meta’s contractual necessity claim', 'relationship': 'MENTIONS', 'to': 'Norway seeks to extend ban on Meta’s
consentless tracking ads across the EU'}, {'from': 'ruling on Meta’s legitimate interests claim', 'relationship': 'MENTIONS', 'to': 'Norway seeks to extend ban
on Meta’s consentless tracking ads across the EU'}, {'from': 'Court ruling to evict Palestinian families from Sheikh Jarrah', 'relationship': 'MENTIONS', 'to':
'Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'privacy complaint lodged against Meta',
'relationship': 'MENTIONS', 'to': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'},
{'from': 'privacy complaint lodged against Meta', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'monthly subscription for an ad-free version of
Meta’s social networks', 'relationship': 'MENTIONS', 'to': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it
‘unfair’ and ‘illegal’'}, {'from': 'monthly subscription for an ad-free version of Meta’s social networks', 'relationship': 'MADE', 'to': 'Meta'}, {'from':
'consumer groups’ complaints about Meta', 'relationship': 'MENTIONS', 'to': 'European consumer groups band together to fight Meta’s self-serving ad-free sub —
branding it ‘unfair’ and ‘illegal’'}, {'from': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and
‘illegal’', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it
‘unfair’ and ‘illegal’', 'relationship': 'MENTIONS', 'to': 'privacy complaint lodged against Meta'}, {'from': 'European consumer groups band together to fight
Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’', 'relationship': 'MENTIONS', 'to': 'monthly subscription for an ad-free version of Meta’s
social networks'}, {'from': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’',
'relationship': 'MENTIONS', 'to': 'consumer groups’ complaints about Meta'}, {'from': 'European consumer groups band together to fight Meta’s self-serving
ad-free sub — branding it ‘unfair’ and ‘illegal’', 'relationship': 'MENTIONS', 'to': 'Meta ad-free subscription'}, {'from': 'European consumer groups band
together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’', 'relationship': 'MENTIONS', 'to': 'Meta’s data protection compliance'},
{'from': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’', 'relationship': 'MENTIONS',
'to': 'Meta’s ‘pay or okay’ model'}, {'from': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and
‘illegal’', 'relationship': 'MENTIONS', 'to': 'Meta’s ad-free subscription'}, {'from': 'Meta ad-free subscription', 'relationship': 'MENTIONS', 'to': 'European
consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'}, {'from': 'Meta ad-free subscription',
'relationship': 'MADE', 'to': 'Meta'}, {'from': 'Meta’s data protection compliance', 'relationship': 'MENTIONS', 'to': 'European consumer groups band together
to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'}, {'from': 'Meta’s ‘pay or okay’ model', 'relationship': 'MENTIONS', 'to':
'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'}, {'from': 'Meta’s ad-free subscription',
'relationship': 'MENTIONS', 'to': 'European consumer groups band together to fight Meta’s self-serving ad-free sub — branding it ‘unfair’ and ‘illegal’'},
{'from': 'Meta’s ad-free subscription', 'relationship': 'MADE', 'to': 'Meta'}, {'from': 'lawsuit against Meta over harms to teens and young users',
'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'}, {'from': 'lawsuit against Meta over
harms to teens and young users', 'relationship': 'PARTICIPATED_IN', 'to': 'Meta'}, {'from': 'harms to teens and young users', 'relationship': 'MENTIONS', 'to':
'Meta seeks legislation that would require parents to approve teens’ app downloads'}, {'from': 'harms to teens and young users', 'relationship': 'ACCUSED_OF',
'to': 'Meta'}, {'from': 'parental approval for teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to
approve teens’ app downloads'}, {'from': 'parental approval for teens’ app downloads', 'relationship': 'SAID', 'to': 'Meta'}, {'from': 'parental responsibility
for approving teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'},
{'from': 'body image issues among teens', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app
downloads'}, {'from': 'app stores', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'},
{'from': 'parental approval of teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app
downloads'}, {'from': 'parental approval of teens’ app downloads', 'relationship': 'SAID', 'to': 'Meta'}, {'from': 'teen mental health', 'relationship':
'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'}, {'from': 'lack of teen protection', 'relationship':
'MENTIONS', 'to': 'Meta seeks legislation that would require parents to approve teens’ app downloads'}, {'from': 'Meta seeks legislation that would require
parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Meta seeks legislation that would require parents to approve
teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'lawsuit against Meta over harms to teens and young users'}, {'from': 'Meta seeks legislation that
would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'harms to teens and young users'}, {'from': 'Meta seeks legislation
that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'parental approval for teens’ app downloads'}, {'from': 'Meta
seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'parental responsibility for approving teens’
app downloads'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'body image
issues among teens'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'app
stores'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'parental approval of
teens’ app downloads'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'teen
mental health'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'lack of teen
protection'}, {'from': 'Meta seeks legislation that would require parents to approve teens’ app downloads', 'relationship': 'MENTIONS', 'to': 'teen safety'},
{'from': 'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert', 'relationship': 'MENTIONS', 'to': 'parents'}, {'from': 'The 5
biggest mistakes parents make at Christmastime, according to a parenting expert', 'relationship': 'MENTIONS', 'to': 'parenting expert'}, {'from': 'The 5 biggest
mistakes parents make at Christmastime, according to a parenting expert', 'relationship': 'MENTIONS', 'to': 'parenting'}, {'from': 'parents', 'relationship':
'MENTIONS', 'to': 'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert'}, {'from': 'parenting expert', 'relationship':
'MENTIONS', 'to': 'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert'}, {'from': "Arnold Schwarzenegger, Reese Witherspoon
parent with tough love: 'Learning from failure'", 'relationship': 'MENTIONS', 'to': 'parental expectations'}, {'from': "Arnold Schwarzenegger, Reese Witherspoon
parent with tough love: 'Learning from failure'", 'relationship': 'MENTIONS', 'to': 'L.A. Parent'}, {'from': "Arnold Schwarzenegger, Reese Witherspoon parent
with tough love: 'Learning from failure'", 'relationship': 'MENTIONS', 'to': 'parenting style'}, {'from': "Arnold Schwarzenegger, Reese Witherspoon parent with
tough love: 'Learning from failure'", 'relationship': 'MENTIONS', 'to': 'parenting'}, {'from': 'parental expectations', 'relationship': 'MENTIONS', 'to':
"Arnold Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure'"}, {'from': 'L.A. Parent', 'relationship': 'MENTIONS', 'to': "Arnold
Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure'"}, {'from': 'parenting style', 'relationship': 'MENTIONS', 'to': "Arnold
Schwarzenegger, Reese Witherspoon parent with tough love: 'Learning from failure'"}, {'from': 'Palestinians', 'relationship': 'MENTIONS', 'to': 'Meta has a
moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices'}, {'from': 'Facebook', 'relationship': 'MENTIONS', 'to': 'How the OpenAI
fiasco could bolster Meta and the ‘open AI’ movement'}, {'from': 'Facebook', 'relationship': 'MENTIONS', 'to': 'Earnings week ahead: Amazon, Microsoft,
Alphabet, Meta, Ford, GM and more'}, {'from': 'Facebook', 'relationship': 'OWNS', 'to': 'Meta'}, {'from': 'Facebook', 'relationship': 'PART_OF', 'to': 'Meta'},
{'from': 'Meta’s open source AI approach', 'relationship': 'MENTIONS', 'to': 'Meta and IBM form an AI Alliance, but to what end?'}, {'from': 'Meta and IBM form
an AI Alliance, but to what end?', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Meta and IBM form an AI Alliance, but to what end?', 'relationship':
'MENTIONS', 'to': 'Meta’s open source AI approach'}, {'from': 'multi-state lawsuit against Meta', 'relationship': 'MENTIONS', 'to': 'Meta turned a blind eye to
kids on its platforms for years, unredacted lawsuit alleges'}, {'from': 'Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit
alleges', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges',
'relationship': 'MENTIONS', 'to': 'multi-state lawsuit against Meta'}, {'from': 'Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit
alleges', 'relationship': 'MENTIONS', 'to': 'verifiable parental consent'}, {'from': 'Meta turned a blind eye to kids on its platforms for years, unredacted
lawsuit alleges', 'relationship': 'MENTIONS', 'to': '2017 Teens Strategic Focus'}, {'from': 'Meta turned a blind eye to kids on its platforms for years,
unredacted lawsuit alleges', 'relationship': 'MENTIONS', 'to': "Turning a blind eye to kids on Meta's platforms"}, {'from': 'Meta turned a blind eye to kids on
its platforms for years, unredacted lawsuit alleges', 'relationship': 'MENTIONS', 'to': "kids on Meta's platforms"}, {'from': 'verifiable parental consent',
'relationship': 'MENTIONS', 'to': 'Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges'}, {'from': 'verifiable parental
consent', 'relationship': 'ACCUSED_OF', 'to': 'Meta'}, {'from': '2017 Teens Strategic Focus', 'relationship': 'MENTIONS', 'to': 'Meta turned a blind eye to kids
on its platforms for years, unredacted lawsuit alleges'}, {'from': "Turning a blind eye to kids on Meta's platforms", 'relationship': 'MENTIONS', 'to': 'Meta
turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges'}, {'from': "Turning a blind eye to kids on Meta's platforms", 'relationship':
'ACCUSED_OF', 'to': 'Meta'}, {'from': "kids on Meta's platforms", 'relationship': 'MENTIONS', 'to': 'Meta turned a blind eye to kids on its platforms for years,
unredacted lawsuit alleges'}, {'from': "kids on Meta's platforms", 'relationship': 'ACCUSED_OF', 'to': 'Meta'}, {'from': "Meta's AI-powered image generator",
'relationship': 'MADE', 'to': 'Meta'}, {'from': 'Imagine with Meta AI', 'relationship': 'MADE', 'to': 'Meta'}, {'from': 'Meta says it’s prioritizing
livestreaming checks during Israel-Hamas war', 'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'Meta Platforms', 'relationship': 'MENTIONS', 'to': 'Earnings
week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more'}, {'from': 'Meta Platforms stock', 'relationship': 'MENTIONS', 'to': 'Earnings week ahead:
Amazon, Microsoft, Alphabet, Meta, Ford, GM and more'}, {'from': 'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'relationship':
'MENTIONS', 'to': 'Meta'}, {'from': 'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'relationship': 'MENTIONS', 'to': 'Facebook'},
{'from': 'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'relationship': 'MENTIONS', 'to': 'Meta Platforms'}, {'from': 'Earnings
week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'relationship': 'MENTIONS', 'to': 'Meta Platforms stock'}, {'from': 'Meta View app',
'relationship': 'MENTIONS', 'to': 'The Ray-Ban Meta smart glasses actually make the future look cool'}, {'from': 'Meta AI', 'relationship': 'MENTIONS', 'to':
'The Ray-Ban Meta smart glasses actually make the future look cool'}, {'from': 'The Ray-Ban Meta smart glasses actually make the future look cool',
'relationship': 'MENTIONS', 'to': 'Meta'}, {'from': 'The Ray-Ban Meta smart glasses actually make the future look cool', 'relationship': 'MENTIONS', 'to': 'Meta
View app'}, {'from': 'The Ray-Ban Meta smart glasses actually make the future look cool', 'relationship': 'MENTIONS', 'to': 'Meta AI'}, {'from': 'The Ray-Ban
Meta smart glasses actually make the future look cool', 'relationship': 'MENTIONS', 'to': 'Ray-Ban Meta Smart Glasses'}, {'from': 'Teenage Engineering',
'relationship': 'MENTIONS', 'to': "Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool"}, {'from': "Teenage Engineering's K.O. II
sampler proves the company can do cost-friendly cool", 'relationship': 'MENTIONS', 'to': 'Teenage Engineering'}, {'from': 'Ray-Ban Meta Smart Glasses',
'relationship': 'MENTIONS', 'to': 'The Ray-Ban Meta smart glasses actually make the future look cool'}, {'from': 'Teen Safety Assist', 'relationship':
'MENTIONS', 'to': 'Discord’s latest teen safety blitz starts with content filters and automated warnings'}, {'from': 'Discord’s latest teen safety blitz starts
with content filters and automated warnings', 'relationship': 'MENTIONS', 'to': 'Teen Safety Assist'}, {'from': 'Discord’s latest teen safety blitz starts with
content filters and automated warnings', 'relationship': 'MENTIONS', 'to': 'teen safety'}, {'from': 'teen safety', 'relationship': 'MENTIONS', 'to': 'Meta seeks
legislation that would require parents to approve teens’ app downloads'}, {'from': 'teen safety', 'relationship': 'MENTIONS', 'to': 'Discord’s latest teen
safety blitz starts with content filters and automated warnings'}, {'from': 'teens', 'relationship': 'MENTIONS', 'to': 'Meta to offer ad-free subscription in
Europe in bid to keep tracking other users'}, {'from': 'parenting', 'relationship': 'MENTIONS', 'to': 'The 5 biggest mistakes parents make at Christmastime,
according to a parenting expert'}, {'from': 'parenting', 'relationship': 'MENTIONS', 'to': "Arnold Schwarzenegger, Reese Witherspoon parent with tough love:
'Learning from failure'"}, {'from': 'parenting', 'relationship': 'MENTIONS', 'to': 'Walks, tech and protein: how to parent your own parents'}, {'from': 'Walks,
tech and protein: how to parent your own parents', 'relationship': 'MENTIONS', 'to': 'parenting'}, {'from': 'Walks, tech and protein: how to parent your own
parents', 'relationship': 'MENTIONS', 'to': 'Self-parenting'}, {'from': 'Self-parenting', 'relationship': 'MENTIONS', 'to': 'Walks, tech and protein: how to
parent your own parents'}, {'from': 'Meta to offer ad-free subscription in Europe in bid to keep tracking other users', 'relationship': 'MENTIONS', 'to':
'Meta'}, {'from': 'Meta to offer ad-free subscription in Europe in bid to keep tracking other users', 'relationship': 'MENTIONS', 'to': 'teens'}]
</item 3>

<item 4>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
His argument essentially was that the improvements and safety features Meta was implementing were not enough and not effective.

However, a policy position like the one Meta announced today is not something that would have been rushed out in a matter of days as a PR response to Bejar’s
claims, but something Meta has been stewing on for some time. After all, the app stores already have age guidelines, so it would seem to follow that they should
also enforce them, if enforcement is to be regulated — or that seems to be Meta’s thinking.

If anything, the call to bring the app stores into the fight is a response to the lack of action in Congress to pass online child safety regulations, like the
hotly debated KOSA (the Kids Online Safety Act) leaving U.S. states to individually create laws due to a void of federal guidelines.

Utah, for instance, passed a measure that will require social networks to give Utah parents access to their children’s posts, messages and comments, and would
block social media access during certain hours.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Bejar, app stores, KOSA, online child safety regulations, social networks, parental approval of teens’ app downloads, age
guidelines, Meta seeks legislation that would require parents to approve teens’ app downloads, Congress, Utah, U.S.
</item 4>

<item 5>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Antigone Davis, coalition of 42 states and D.C., lawsuit against Meta over harms to teens and young users, Frances Haugen, harms to
teens and young users, parental approval for teens’ app downloads, parental responsibility for approving teens’ app downloads, Meta seeks legislation that would
require parents to approve teens’ app downloads, Pew Research
</item 5>

<item 6>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The governor also signed a bill to prevent social media companies from adding features designed to addict minors to their platforms. A judge in Arkansas
recently blocked a law that would have required parental permission for minors to create social media accounts.

Beyond Meta, Montana also became the first state to ban TikTok, claiming it was being used by the Chinese Communist Party using TikTok to spy on Americans. But
Meta’s threat, in terms of teen usage, isn’t one of a foreign power collecting private data on users, but rather risks to teen mental health, body comparison
issues and exposure to sexual predators. Montana could easily turn its eyes to Meta next, after the TikTok bill went through.

The overhead of managing apps to meet individual state regulations would be a headache for Meta, but Davis also argues it would lead to a lack of teen
protection, in some cases.

“U.S.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Arkansas
- Northern Lights: Here are the best tips to help you spot the stunning display in the US and abroad -MENTIONS-> Arkansas
- Pittsburgh boy, 10, needs second liver transplant to save his life: ‘Only possible through love’ -MENTIONS-> Arkansas
- Helena World Chronicle -LOCATED_IN-> Arkansas
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> Chinese Communist Party

Entities: Arkansas, Chinese Communist Party, Meta, TechCrunch, sexual predators, teen mental health, body comparison issues, lack of teen protection, Meta seeks
legislation that would require parents to approve teens’ app downloads, Davis, TikTok, U.S., Montana
</item 6>

<item 7>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
“U.S. states are passing a patchwork of different laws, many of which require teens (of varying ages) to get their parent’s approval to use certain apps, and
for everyone to verify their age to access them,” she wrote. “Teens move interchangeably between many websites and apps, and social media laws that hold
different platforms to different standards in different states will mean teens are inconsistently protected,” Davis said.

Meta’s proposed “better way” of requiring parents to manage a teen’s app downloads pushes the burden back to the app stores and to parents, who often today do
not monitor or manage their teens’ smartphone use. But while it’s true that the app stores and smartphone makers would know the ages of their users, requiring
parents to approve apps is the equivalent of overriding parental discretion and choice.

Other entertainment ratings — like those for TV, movies or video games — are meant to be guidelines for parents, not laws. Meta’s position that apps are somehow
different, then, is a fairly extreme position to take.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> legislation
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> legislation
- Meta -MENTIONS-> Colombia
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp

Entities: legislation, Meta, TechCrunch, app stores, Meta seeks legislation that would require parents to approve teens’ app downloads, U.S. states, Social
media laws, Entertainment ratings, Video games, movies, TV, Davis, smartphones, apps
</item 7>

<item 8>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Most recently, Instagram users accused Meta of “shadowbanning” their Stories about Palestine.

It’s the latest in a lengthy history of incidents on Meta platforms that reflect an inherent bias against Palestinian users in its processes, as documented by
years of complaints from both inside and outside the company. The company may not intentionally suppress specific communities, but its moderation practices
often disproportionately affect Palestinian users.

For instance, Meta struggles to navigate the cultural and linguistic nuances of Arabic, a language with over 25 dialects, and has been criticized for neglecting
to adequately diversify its language resources. The company’s black-and-white policies often preclude it from effectively moderating any nuanced topic, like
content that discusses violence without condoning it. Advocacy groups have also raised concerns that Meta’s partnerships with government agencies, such as the
Israeli Cyber Unit, politically influence the platform’s policy decisions.

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> moderation
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> EU
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> shadowbanning
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Stone
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Threads

Entities: shadowbanning, Israeli Cyber Unit, moderation bias, Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices, Meta,
TechCrunch, violence, Arabic, Instagram, Palestine
</item 8>

<item 9>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta’s position that apps are somehow different, then, is a fairly extreme position to take.

“Instead of designing its products with kids’ safety and privacy in mind, Meta would rather push blame onto parents and other companies to obscure the damage
they’ve already done to young people,” said Sacha Haworth, executive director of the Tech Oversight Project, a nonprofit devoted to reigning in Big Tech, in
reaction to the news. “No one disputes that Google and Apple need to do more to protect children on their platforms, apps and devices. That fact does not
absolve Mark Zuckerberg or Meta for years of willful negligence, and this announcement shows that the company’s outright hostility toward protecting kids is
sewn into the fabric of its profit model.”

Apple and Google did not return requests for comment.

Updated, 11/15/23, 4:40 PM ET, with comment from Tech Oversight Project.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Mark Zuckerberg
- Want to help your kids achieve their potential? This expert has some advice -MENTIONS-> Mark Zuckerberg
- Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges -MENTIONS-> Mark Zuckerberg
- Every movie and show coming to Netflix in November -MENTIONS-> Mark Zuckerberg

Entities: Mark Zuckerberg, Apple, Meta, TechCrunch, Meta seeks legislation that would require parents to approve teens’ app downloads, Sacha Haworth, Tech
Oversight Project, willful negligence, children’s safety and privacy, Google, big tech
</item 9>

<item 10>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Arturo Bejar, body image issues among teens, sexual predators, unwanted sexual advances and harassment, Adam Mosseri hauled before
Congress, Meta seeks legislation that would require parents to approve teens’ app downloads, Congress, Instagram, teen safety, Adam Mosseri
</item 10>
=== END ITEM id=805b5fdcc898390b ===
