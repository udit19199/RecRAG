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

This file has 3 items.

=== ITEM id=933f82514b5fed0c ===
Gold sentences and facts:
Sentence 0: Halloween isn’t the only holiday on the horizon now that October is drawing to a close.
  Fact 0.0: Halloween is a holiday on the horizon.
  Fact 0.1: Halloween is not the only holiday on the horizon.
  Fact 0.2: October is drawing to a close.
  Fact 0.3: Another holiday is on the horizon now that October is drawing to a close.
Sentence 1: If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
  Fact 1.0: A reader may be in the midst of a new relationship this holiday season.
  Fact 1.1: A reader in a new relationship this holiday season may be faced with the opportunity to introduce their significant other to their family.

Retrieved items:
<item 1>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
However, it’s likely you’ve found yourself mulling over an array of reasons why the holidays may not be the right time for such an introduction.

The holiday season is often perceived as the most wonderful time of year, but that’s not necessarily the case for everyone.

Graph facts:
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Family
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Family
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Family
- The best comedies you’ll find on streaming right now -MENTIONS-> Family
- Jada Pinkett Smith -SAID-> Family

Entities: Family, Should you introduce your partner to your family during the holidays? Relationship experts weigh in, relationships, holiday season, Lifestyle
Edit, The Independent
</item 1>

<item 2>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I think anytime that emotions are heightened, with the holidays certainly being one of those times, our expectations are really high,” she said. “We’re
expecting it to be this joyful, happy, perfect occasion. So, we sometimes forget that the holidays come with a lot of stress.”

Despite tensions between relatives at the dinner table, that doesn’t mean your relationship shouldn’t be celebrated. Saba Lurie is a licensed marriage and
family therapist at Take Root Thereapy in California. She explained to The Independent that while there isn’t a right or wrong time to introduce your partner to
your family, the intentions of such an introduction must be clear. Not only should you have already discussed where your relationship is headed, but you should
also tell your partner everything they need to know about your family.

“I think you should tell the person why you’re bringing them home for the holidays, so you’re on the same page about the intentions and what’s motivating the
decision to include,” she said.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Saba Lurie, Take Root Thereapy, California, The
Independent
</item 2>

<item 3>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Blink Video Doorbell (with Blink Outdoor 4) $ 90 $ 180 50 % off $ 90 $ 90 $ 180 50 % off Blink’s Video Doorbell is the best if you’re looking for a
budget-friendly buzzer that offers motion-activated recording and alerts, night vision, two-way audio, and up to two years of battery life. $90 at Amazon

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Some more deals to kickstart your weekend

Graph facts:
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Gmail
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> LED Strip Light M1
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> peppermint mochas
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> 2023 holiday season
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Govee M1 Matter Compatible LED Strip Light

Entities: Govee’s colorful LED light strips are on sale just in time for Halloween, colorful LED light strips, Blink Video Doorbell, Blink Outdoor 4, X, Verge
Deals, Govee, Amazon, The Verge
</item 3>

<item 4>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Zar, Getty Images, The Independent
</item 4>

<item 5>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
That isn’t necessarily a cause for concern, as long as you inform your significant other about the interworkings of your family dynamics.

“I think that we tend to be our best relational self when we’ve really done the work around understanding all of our past relationships,” Zar explained. “You
can tell your partner: ‘I don’t have a close relationship with my brother, and here’s why. Here are the intentional choices that I’ve made around it.’ Or you
can say: ‘Here’s my chosen family. These are the people who are important to me.’ Your significant other will see that you’ve been introspective and have
takeaways around how you’ve navigated your family’s origin.”

As long as you and your partner have discussed your relationship expectations - both before and during the holidays – introducing them to your family may deepen
your relationship and could be a step in the right direction.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Zar, family dynamics, past relationships,
relationship expectations, chosen family, The Independent, holidays
</item 5>

<item 6>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Take Root Thereapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Saba Lurie
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> American
Psychological Association

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Rachel Zar, Spark Chicago Therapy, Hanukkah, New
Years, family problems, American Psychological Association, Christmas, The Independent, stress
</item 6>

<item 7>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
publication_date: 2023-11-27T08:45:59+00:00
publisher: None
section: None
subjects: [{'name': 'Echo', 'type': '__KGBuilder__'}, {'name': 'Cyber Weekend', 'type': '__KGBuilder__'}, {'name': 'Echo Show', 'type': '__KGBuilder__'},
{'name': 'smart screen', 'type': '__KGBuilder__'}, {'name': 'smart lights', 'type': '__KGBuilder__'}, {'name': 'Audible Premium Plus', 'type': '__KGBuilder__'},
{'name': 'Echo smart displays', 'type': '__KGBuilder__'}, {'name': 'Echo smart speakers', 'type': '__KGBuilder__'}, {'name': 'Fire TVs', 'type':
'__KGBuilder__'}, {'name': 'MGM+', 'type': '__KGBuilder__'}, {'name': 'Liquid Retina display', 'type': '__KGBuilder__'}, {'name': 'A14 Bionic chip', 'type':
'__KGBuilder__'}, {'name': 'USB-C charging', 'type': '__KGBuilder__'}, {'name': 'Microsoft Surface Laptop Go 3', 'type': '__KGBuilder__'}, {'name': 'Surface Go
2', 'type': '__KGBuilder__'}, {'name': 'Go 3', 'type': '__KGBuilder__'}, {'name': 'QuietComfort 45s', 'type': '__KGBuilder__'}, {'name': "Amazon's Cyber Monday
sale", 'type': '__KGBuilder__'}, {'name': 'META50', 'type': '__KGBuilder__'}, {'name': "Mashable's daily Deals newsletter", 'type': '__KGBuilder__'}, {'name':
'Terms of Use', 'type': '__KGBuilder__'}, {'name': 'Tablet deals', 'type': '__KGBuilder__'}, {'name': 'Gaming keyboards', 'type': '__KGBuilder__'}, {'name': 'VR
headsets', 'type': '__KGBuilder__'}, {'name': 'Vacuum deals', 'type': '__KGBuilder__'}, {'name': 'Cordless vacuums', 'type': '__KGBuilder__'}, {'name': 'Robot
vacuums and vacuum/mop hybrids', 'type': '__KGBuilder__'}, {'name': 'Kitchen deals', 'type': '__KGBuilder__'}, {'name': 'Pizza ovens', 'type': '__KGBuilder__'},
{'name': 'Sparkling water makers', 'type': '__KGBuilder__'}, {'name': 'Toaster ovens', 'type': '__KGBuilder__'}, {'name': 'Amazon device and service deals',
'type': '__KGBuilder__'}, {'name': 'Blink', 'type': '__KGBuilder__'}, {'name': 'Chromecast', 'type': '__KGBuilder__'}, {'name': 'Google Nest', 'type':
'__KGBuilder__'}, {'name': 'Fitness deals', 'type': '__KGBuilder__'}, {'name': 'Smartwatches and fitness trackers', 'type': '__KGBuilder__'}, {'name': 'Apple
deals', 'type': '__KGBuilder__'}, {'name': 'Beauty tech deals', 'type': '__KGBuilder__'}, {'name': 'Drone deals', 'type': '__KGBuilder__'}, {'name': 'Camera
deals', 'type': '__KGBuilder__'}, {'name': 'GoPro', 'type': '__KGBuilder__'}, {'name': 'Lego deals', 'type': '__KGBuilder__'}, {'name': 'Gift card deals',
'type': '__KGBuilder__'}, {'name': 'Mac', 'type': '__KGBuilder__'}, {'name': 'Microsoft', 'type': '__KGBuilder__'}, {'name': 'Apple', 'type': '__KGBuilder__'},
{'name': 'AirPods', 'type': '__KGBuilder__'}, {'name': 'AirPods Max', 'type': '__KGBuilder__'}, {'name': 'Echo devices', 'type': '__KGBuilder__'}, {'name':
'Alexa', 'type': '__KGBuilder__'}, {'name': 'privacy policy', 'type': '__KGBuilder__'}, {'name': 'iPad', 'type': '__KGBuilder__'}, {'name': 'Eero', 'type':
'__KGBuilder__'}, {'name': 'Meta Quest 2', 'type': '__KGBuilder__'}, {'name': 'Apple Watch', 'type': '__KGBuilder__'}, {'name': 'Smart Home Deals', 'type':
'__KGBuilder__'}, {'name': 'Speaker Deals', 'type': '__KGBuilder__'}, {'name': 'Bose', 'type': '__KGBuilder__'}, {'name': 'Earbuds', 'type': '__KGBuilder__'},
{'name': 'MacBooks', 'type': '__KGBuilder__'}, {'name': 'MacBook Air', 'type': '__KGBuilder__'}, {'name': 'ANC', 'type': '__KGBuilder__'}, {'name': 'Echo Buds',
'type': '__KGBuilder__'}, {'name': 'MacBook', 'type': '__KGBuilder__'}, {'name': 'headphones', 'type': '__KGBuilder__'}, {'name': 'Fire TV', 'type':
'__KGBuilder__'}, {'name': 'Black Friday', 'type': '__KGBuilder__'}, {'name': 'Kindle', 'type': '__KGBuilder__'}, {'name': 'Prime', 'type': '__KGBuilder__'},
{'name': 'Gaming mice', 'type': '__KGBuilder__'}, {'name': 'Black Friday sale', 'type': '__KGBuilder__'}, {'name': 'Intel', 'type': '__KGBuilder__'}, {'name':
'Amazon', 'type': '__KGBuilder__'}, {'name': 'Cyber Monday sale', 'type': '__KGBuilder__'}, {'name': 'Intel Core i5 processor', 'type': '__KGBuilder__'},
{'name': 'Fire tablets', 'type': '__KGBuilder__'}, {'name': 'gaming deals', 'type': '__KGBuilder__'}, {'name': 'Amazon credit', 'type': '__KGBuilder__'},
{'name': 'gaming headsets', 'type': '__KGBuilder__'}, {'name': 'Apple Watches', 'type': '__KGBuilder__'}]
</item 7>

<item 8>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Halloween isn’t the only holiday on the horizon now that October is drawing to a close. Once the clock strikes midnight on October 31st, Mariah Carey will
defrost, peppermint mochas will inevitably return, and the 2023 holiday season will have officially begun.

Thankfully for those looking to get in the festive spirit, Govee’s Matter-ready LED Strip Light M1 is on sale for an all-time low of $44.99 ($15 off) at Amazon
when you clip the on-page coupon. You can also buy it for $3 more direct from Govee through October 31st, which is still one of its better prices to date.

While handy throughout the year, Govee’s 6.56-foot (2m) light strip is particularly convenient during Halloween and other holidays. Not only can the bright,
tunable LEDs flash and blink to the beat of your music, but they can also act as a festive piece of decor that adds a pop of color to your holiday parties.

Graph facts:
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Google flagship wireless earbuds
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Matter protocol
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Gmail
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Govee M1 Matter Compatible LED Strip Light
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Govee LED light strips

Entities: LED Strip Light M1, peppermint mochas, 2023 holiday season, Govee’s colorful LED light strips are on sale just in time for Halloween, Halloween,
Mariah Carey, Govee, Amazon, The Verge
</item 8>

<item 9>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
What’s particularly great about this strip, though, is that it supports the newer Matter protocol, which allows you to control the lights with a wide array of
smart home platforms.

Govee LED Strip Light M1 Matter $ 45 $ 60 25 % off $ 45 $ 45 $ 60 25 % off The customizable Govee M1 Matter Compatible LED Strip Light is 6.56 feet long and can
display millions of colors and shades of white. You can also use it to create light patterns that sync with music. $45 at Amazon$48 at Govee

Though they launched last year, Google’s flagship wireless earbuds continue to impress us with their terrific sound quality and good noise cancellation, which
is why they’re one of our favorite pairs of wireless earbuds. We’re also fans because of all the other perks they offer, like multipoint connectivity, long
battery life, and a good transparency mode.

Graph facts:
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> peppermint mochas
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> 2023 holiday season
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Gmail
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> LED Strip Light M1
- Govee’s colorful LED light strips are on sale just in time for Halloween -MENTIONS-> Govee LED light strips

Entities: Govee M1 Matter Compatible LED Strip Light, Google flagship wireless earbuds, Matter protocol, Govee’s colorful LED light strips are on sale just in
time for Halloween, smart home platforms, Google, Halloween, Govee, Amazon, music, The Verge
</item 9>

<item 10>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I’d recommend pulling your partner aside and saying: ‘How are we doing?’ Like a temperature check,” she added. “Those are points where the two of you can come
and remind each other: ‘We are a team in this.’ Or, the two of you could go on a walk and separate from the family, just for a little bit of time to reset
within the relationship.”

During the holiday season, your partner may be seeing how you interact with your family for the first time. But that’s not necessarily a bad thing, according to
the relationship experts. “I think introducing a significant other to family is a really vulnerable act, and that vulnerability is going to be what allows for
greater intimacy,” Lurie said. “When your significant other sees you with your family, they may see another side of you that they may not have had access to
before.”

(Getty Images)

On the other hand, your partner could learn that your relationship with your parents and extended family members is complex.

Graph facts:
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Family
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Family
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Family
- The best comedies you’ll find on streaming right now -MENTIONS-> Family
- Jada Pinkett Smith -SAID-> Family

Entities: Family, Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Intimacy, Lurie, Getty Images,
relationships, The Independent
</item 10>
=== END ITEM id=933f82514b5fed0c ===

=== ITEM id=4709909e3368deae ===
Gold sentences and facts:
Sentence 0: In July, speaking on the podcast he shares with his brother, New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City.
  Fact 0.0: In July, Kelce spoke on the podcast New Heights, which he shares with his brother.
  Fact 0.1: In July, Kelce said he intended to give Swift a friendship bracelet with his number on it.
  Fact 0.2: Kelce intended to give the friendship bracelet to Swift during her Eras Tour concert in Kansas City.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: On his podcast, Kelce later invited Swift, the singer of 'Anti-Hero', to one of his games at Arrowhead Stadium.
  Fact 1.1: Swift famously accepted Kelce's invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Jason, Taylor Swift, rumored romance, Travis Kelce, pop cultural moment, Swift-Kelce coverage, Taylor Swift-Travis Kelce timeline: Everything to
know about rumored romance between pop star, Chiefs TE, CBSSports.com
</item 1>

<item 2>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.

Graph facts:
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> NFL
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Arrowhead Stadium
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Kansas City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Eras Tour concert
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars

Entities: When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Andy Reid, Vulture, New York City, visit to New York City, re-recording
1989 tracks, quietly hanging out, Taylor Swift, Travis Kelce, The Age, Chiefs
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.

Graph facts:
- Taylor Swift -MENTIONS-> jersey sales
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Taylor Swift

Entities: Taylor Swift, rumored romance, Travis Kelce, Belichick, Front Office Sports, Apple, Stubhub, Taylor Swift-Travis Kelce timeline: Everything to know
about rumored romance between pop star, Chiefs TE, Chiefs game, CBSSports.com, Chiefs, Instagram, New England Patriots
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.

Graph facts:
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Eras Tour, Travis Kelce, Taylor Swift reveals secret start to her relationship with Travis Kelce, The Independent
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> All Too Well
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> All Too Well
- ‘It happens’: All Blacks defend star centre after spray as Cheika heaps pressure on ‘favourites’ New Zealand -MENTIONS-> Argentina
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Argentina
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Argentina

Entities: All Too Well, Argentina, Buenos Aires, Karma, Time, Taylor Swift, Travis Kelce, Chiefs, Taylor Swift reveals secret start to her relationship with
Travis Kelce, Eras Tour show, Style, The Independent
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 6>

<item 7>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Could they have been driving off into the sunset? Of course, none of this actually confirms Swift and Kelce are actually dating. But regardless of whether it’s
a friendship or a romantic relationship, it has sent shockwaves through two massive cultural franchises, expanding the NFL’s audience into uncharted territory.
It’s a cultural pairing on par with – and perhaps even bigger than – Kim Kardashian and Kanye West.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover, Kelce, Kanye West, Taylor Swift, Kim Kardashian, The Age
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."

Graph facts:
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift
- Taylor Swift -MENTIONS-> jersey sales

Entities: Taylor Swift, Arrowhead, rumored romance, Travis Kelce, Mahomes, postgame press conference, Swiftie talk, Taylor Swift-Travis Kelce timeline:
Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, game, Donna Kelce
</item 8>

<item 9>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Jason Kelce
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Jason Kelce
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> Jason Kelce
- Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jason Kelce
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Jason Kelce

Entities: Jason Kelce, Taylor Swift, Travis Kelce, New York Jets, The Independent, Kansas City Chiefs, National Football League, game against the New York Jets,
New Heights with Jason and Travis Kelce, Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Chicago Bears
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chicago Bears
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Chicago Bears
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> Chicago Bears
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Chicago Bears

Entities: Chicago Bears, Taylor Swift, Travis Kelce, Chiefs' Week 3 game against the Chicago Bears, nonprofit charity event, Vigilante Sh--, Fifteen, Taylor
Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE, CBSSports.com, Chiefs, Donna Kelce
</item 10>
=== END ITEM id=4709909e3368deae ===

=== ITEM id=c2bf4e453455154e ===
Gold sentences and facts:
Sentence 0: This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
  Fact 0.0: This is not the first time Jada has opened up about her long-term commitment to her husband.
Sentence 1: Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus.
  Fact 1.0: Jada then praised her children for how they have helped her grow.
  Fact 1.1: Jada added: 'My children, they're little gurus.'
Sentence 2: She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep sense of self-acceptance”.
  Fact 2.0: Jada and Will have two children, Jaden, 25, and Willow, 22.
  Fact 2.1: Jada praised Jaden and Willow for being her 'little gurus' throughout this time.
  Fact 2.2: Jada noted that Jaden and Willow have taught her 'a deep sense of self-acceptance'.

Retrieved items:
<item 1>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
During an interview with Hoda Kotb for Today, which aired on 13 October, the actor confirmed that she and her partner have been living “completely separate
lives” for the past seven years. However, according to Jada, their choice to live in separate homes doesn’t mean that they’ll be getting divorced anytime soon.

“I made a promise that there will never be a reason for us to get a divorce,” she said. “We will work through whatever. And I just haven’t been able to break
that promise.”

When asked why she and Will didn’t “just amicably divorce,” Jada simply replied: “We don’t want to.”

She then explained how her family shaped that decision, as she and the King Richard star share two children - Jaden, 25, and Willow, 22. Will also has a
30-year-old son, Trey, who he shares with ex-wife Sheree Zampino. “We love our family and we love each other.
</item 1>

<item 2>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Willow
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Jada Pinkett Smith
- Willow -PARTICIPATED_IN-> Willow’s birth
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Willow
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Jada Pinkett Smith

Entities: Willow, Jada Pinkett Smith, Will Smith, Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith, Jaden, Enoch Pratt Free
Library, press event for Worthy, Worthy, The Independent, Baltimore, Maryland
</item 2>

<item 3>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Less than two weeks later, Jada and Will appeared on an episode of Jada’s Red Table Talk where Jada confirmed that she did have a relationship with Alsina,
saying that she and Will were separated at the time.

“We decided that we were going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,”
Jada said at the time.

“From there, as time went on, I got into a different kind of entanglement with August... Yes, it was a relationship absolutely.”

Jada also said that Will didn’t give permission, as the only person who could give her permission was herself.

2021: Jada discusses her and Will’s sex life, Will’s memoir is released

In another episode of Red Table Talk in October 2021, Jada said it “takes work” to maintain a spark in the bedroom.

“It’s hard... the thing Will and I talk about a lot is the journey.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -PUBLISHED_BY-> The Independent - Life and Style

Entities: separation, A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation, August Alsina, Jada Pinkett Smith, relationship with
August Alsina, permission, maintaining a spark in the bedroom, episode of Red Table Talk, Will Smith, Red Table Talk, The Independent
</item 3>

<item 4>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> 2022 Academy Awards
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> Chris Rock
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Chris Rock
- Chris Rock -PARTICIPATED_IN-> 94th annual Oscars
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> 2022 Academy Awards

Entities: Chris Rock, 2022 Academy Awards, King Richard, separation, A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation, Jada
Pinkett Smith, Will Smith, Girls Trip, The Independent, People
</item 4>

<item 5>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Jada Pinkett Smith, Will Smith, Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith, Interview with
People, The Independent, People
</item 5>

<item 6>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -PUBLISHED_BY-> The Independent - Life and Style

Entities: separation, A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation, Jada Pinkett Smith, Rock, altercation, GI Jane 2, award
for Best Actor, Will Smith, The Independent
</item 6>

<item 7>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> 2022 Academy Awards
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> Chris Rock
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Chris Rock
- Chris Rock -PARTICIPATED_IN-> 94th annual Oscars
- Jada Pinkett Smith reveals why she and Will Smith never signed a prenup -MENTIONS-> 2022 Academy Awards

Entities: Chris Rock, 2022 Academy Awards, separation, self-acceptance, Willow, Jada Pinkett Smith, Jada Pinkett Smith and Will Smith have been separated since
2016 - when should couples just get divorced?, Will Smith, Jaden, Worthy, The Independent - Life and Style, People
</item 7>

<item 8>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“We’ve been doing some really heavy-duty work together,” Jada said. “We just got deep love for each other and we are going to figure out what that looks like
for us.”

The pair have been married since 1997 and share son Jaden, 25, and daughter Willow, 22. The Fresh Prince of Bel-Air star also has a son Trey, 30, whom he shares
with his first wife, Sheree Zampino.

As for her children, Jada praised them for how they’ve helped her grow. “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance,”
she told the outlet.

Jada and Will have often opened up about their relationship over the years, and have had very public ups and downs throughout their marriage. Here’s a timeline
of their relationship.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> separation
- Jacob Lucas -SAID-> separation
- Susan Winter -SAID-> separation
- Winter -SAID-> separation
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> self-acceptance

Entities: separation, Trey, Fresh Prince of Bel-Air, self-acceptance, deep love, Sheree Zampino, A timeline of Will Smith and Jada Pinkett Smith’s relationship
amid separation, Willow, Jada Pinkett Smith, Will Smith, Jaden, The Independent - Life and Style
</item 8>

<item 9>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.

Graph facts:
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Jada Pinkett Smith
- Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> Jada Pinkett Smith
- Jada Pinkett Smith -SAID-> separation
- Jada Pinkett Smith -SAID-> self-acceptance

Entities: Jada Pinkett Smith, Will Smith, interview, prenup, Jada Pinkett Smith reveals why she and Will Smith never signed a prenup, New York Times, memoir,
The Independent
</item 9>

<item 10>
Graph query row:
article: Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced?
published_on: 2023-10-16T13:41:30+00:00
publisher: The Independent
mentioned_people: ['Chris Rock', 'Sheree Zampino', 'Willow', 'August Alsina', 'Jada Pinkett Smith', 'Rock', 'Hoda Kotb', 'Susan Winter', 'Winter', 'Jacob
Lucas', 'Will Smith', 'Alsina', 'Jaden', 'Lucas']
self_acceptance_topics: ['self-acceptance']
said_entities: []
</item 10>
=== END ITEM id=c2bf4e453455154e ===
