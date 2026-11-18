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

=== ITEM id=a0b10759063bb02b ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: The quoted speaker has the Eras Tour booked all through next year.
  Fact 0.1: The quoted speaker would rather die than cancel a show.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker told NPR that it was almost like a very strange, subtle clue to the media.
  Fact 1.1: The speaker told NPR that the clue suggested the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker told NPR that something really major and traumatic can happen to her and the media does not know about it.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.1: Swift's relationship with Kelce is new.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”

Graph facts:
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> WSJ Magazine
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: WSJ Magazine, Taylor Swift, Travis Kelce, Taylor Swift reveals secret start to her relationship with Travis Kelce, The Independent
</item 1>

<item 2>
Graph query row:
article: Taylor Swift is Time’s Person of the Year
published_on: 2023-12-06T13:55:17+00:00
publisher: The Independent
person: Taylor Swift
connection: MENTIONS
page: Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
passage: Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.

Graph facts:
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Anti-Hero
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Anti-Hero
- Taylor Swift -MENTIONS-> Instagram

Entities: Anti-Hero, Time, Taylor Swift, Travis Kelce, relationship, Taylor Swift reveals secret start to her relationship with Travis Kelce, paparazzi, media,
The Independent - Life and Style
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.

Graph facts:
- Taylor Swift is Time’s Person of the Year -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Travis Kelce, Kelce’s game on 3 December, public relationship, Taylor Swift reveals secret start to her relationship with Travis
Kelce, The Independent
</item 4>

<item 5>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”

Graph facts:
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Taylor Swift is Time’s Person of the Year, The Independent
</item 5>

<item 6>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift
- Taylor Swift -MENTIONS-> Instagram
- Taylor Swift -MENTIONS-> jersey sales
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> Taylor Swift

Entities: Time, Time’s 2023 Person of the Year, ticket sales, Taylor Swift, Eras Tour, Taylor Swift is Time’s Person of the Year, The Independent
</item 6>

<item 7>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.

Graph facts:
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Time
- Taylor Swift -MENTIONS-> jersey sales
- Taylor Swift -MENTIONS-> Instagram
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Taylor Swift
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Taylor Swift

Entities: Time, Taylor Swift, Taylor Swift is Time’s Person of the Year, Ana Clara Benevides, Big Machine Records, investment fund, master recordings, Taylor
Swift’s first performance in the country, Scooter Braun acquired Big Machine Records, Sale of the master recordings, Scooter Braun, cardiac arrest, The
Independent - Life and Style
</item 7>

<item 8>
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
</item 8>

<item 9>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”

Graph facts:
- Kelce -MENTIONS-> Raiders
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Kelce
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Kelce
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Kelce
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Kelce

Entities: Kelce, Time, Taylor Swift, Joe Alwyn, Conversations with Friends, Taylor Swift is Time’s Person of the Year, The Independent
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.

Graph facts:
- Taylor Swift -MENTIONS-> jersey sales
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> 1989 (Taylor’s
Version)
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> 1989 (Taylor’s Version)
- Taylor Swift reveals secret start to her relationship with Travis Kelce -MENTIONS-> Time
- Taylor Swift -MENTIONS-> Instagram

Entities: 1989 (Taylor’s Version), Time, Taylor Swift, Taylor Swift is Time’s Person of the Year, Fearless (Taylor’s Version), Red (Taylor’s Version), Speak Now
(Taylor’s Version), tour, Alwyn, Tree Paine, rumours about Taylor Swift and Alwyn’s relationship, Grammy, The Independent
</item 10>
=== END ITEM id=a0b10759063bb02b ===

=== ITEM id=32de29eb1810f1ad ===
Gold sentences and facts:
Sentence 0: The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive thinking.
  Fact 0.0: The pastor explained that Jesus brought more positivity than anyone.
  Fact 0.1: The pastor explained that it is important to keep faith aligned with the practice of positive thinking.
  Fact 0.2: The pastor explained that keeping faith aligned with positive thinking is important because Jesus brought more positivity than anyone.
Sentence 1: He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically.
  Fact 1.0: The pastor added that consistently tapping into the habit makes one's relationship with Jesus grow deeper and deeper.
  Fact 1.1: The pastor added that the relationship with Jesus grows automatically.

Retrieved items:
<item 1>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Bradley said, "Positive thinking is relational, and there is no greater source of joy in our lives than God."
</item 1>

<item 2>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
He said turning to Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.

"The Word of God is living and active, bringing strength, courage, wisdom, and protection," he said.

"God's thoughts and ways are higher than ours, but when we spend time, each day reading His Word and listening to the voice of the Good Shepherd, we are filled
with new thoughts that change the trajectory of our lives."

Bradley also shared his dos and don'ts of positive thinking.

He said it's not "denial and pretending that there are no challenges," he said.

"Positive thinking is not flattery and frivolous compliments."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Rather, positive thinking is realistic, full of truth and love — and aligned with heaven and scripture.

"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating unrealistic expectations for other people and
putting extra stress on them to be perfect."

Graph facts:
- There’s something going on with AI startups in France -MENTIONS-> mental health
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> mental health
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> mental health
- Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes -MENTIONS-> mental health
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> mental health

Entities: mental health, FOX News - Lifestyle, positive thinking, Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in,
Word of God, Good Shepherd, heaven, scripture, Bradley, Jesus, God
</item 2>

<item 3>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
We do not have to believe, harbor or entertain negative first thoughts. We can be proactive by intentionally choosing our second thought."

It is no accident that the pastor's new book, his second, is called "The Power of the Second Thought."

It dives into the power of understanding that the first thoughts we have are not always the best — and that it's wise to take some time and get to the "second
thoughts," which can yield greater understanding.

STORYTELLER REVEALS HIS FAITH AT WORK: ‘I REFUSE TO TRY TO SCREW SOMEONE IN THE NAME OF BUSINESS’

Said Bradley to Fox News Digital in emailed comments this week, "How can we choose a positive attitude and approach during the day? Reject and replace!"

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> Fox News
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> Fox News
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -PUBLISHED_BY-> Fox News
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -PUBLISHED_BY-> Fox News
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -PUBLISHED_BY-> Fox News

Entities: Fox News, Fox News Digital, positive attitude and approach, faith at work, Faith dilemma: How effective is positive thinking when life gets difficult?
Pastor weighs in, The Power of the Second Thought, Bradley
</item 3>

<item 4>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Positive thinking is not creating unrealistic expectations for other people and putting extra stress on them to be perfect."

Bradley said, "Positive thinking is a vital aspect of loving your neighbor, nourishing your mind, discerning between light and darkness, good and evil and right
and wrong."

"Positive thinking leads to taking action and improving situations. Positive thinking brings out the best in the people around you. Positive thinking rejects
excuses, takes responsibility, and overcomes setbacks."

The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive
thinking.

CLICK HERE TO GET THE FOX NEWS APP

"Jesus exudes a trustworthy positivity that is full of meaning, faithfulness, and transformation," he noted.

"God renews our thinking and vision, empowers us to make intentional choices about our focus and perspective and trains us how to win the battle of our minds,"
he added.

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> Fox News
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> Fox News
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -PUBLISHED_BY-> Fox News
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -PUBLISHED_BY-> Fox News
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -PUBLISHED_BY-> Fox News

Entities: Fox News, faith, FOX News app, positive thinking, Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in,
Bradley, Jesus, God
</item 4>

<item 5>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Pastor Jesse Bradley of Auburn, Washington runs Grace Community Church.

He's ministered to many, many people struggling with a variety of faith dilemmas and issues in their lives.

In this month of October 2023, as Americans grapple with a variety of issues, he addressed the notion of positive thinking — and shared with Fox News Digital
what it would "look like to change your thinking and shift in a new direction."

WASHINGTON PASTOR, ONCE GRAVELY ILL, SHARES HOW ‘THE POWER OF THE SECOND THOUGHT’ TRANSFORMED HIS LIFE

Said Pastor Bradley, "Psychologists report that we have in between 6,000 and 60,000 thoughts a day, based on the person and situation. The National Science
Foundation shares that up to 80% of our first thoughts are not helpful, even destructive."

What happens is that "when life is difficult, we can become flooded with thoughts that are drenched in fear, worry, stress, selfishness, impurity, selfishness
and lies.

Graph facts:
- Meta and IBM form an AI Alliance, but to what end? -MENTIONS-> National Science Foundation
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> Fox News
- National Science Foundation -PART_OF-> AI Alliance
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -PUBLISHED_BY-> Fox News
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> Fox News

Entities: National Science Foundation, Fox News, Jesse Bradley, Grace Community Church, Auburn, Washington, positive thinking, faith dilemmas, first thoughts,
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in
</item 5>

<item 6>
Graph query row:
article: Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in
published_on: 2023-10-28T09:00:12+00:00
publisher: Fox News
passages: ['Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle,
2023-10-28T09:00:12+00:00)\nPastor Jesse Bradley of Auburn, Washington runs Grace Community Church.\n\nHe\'s ministered to many, many people struggling with a
variety of faith dilemmas and issues in their lives.\n\nIn this month of October 2023, as Americans grapple with a variety of issues, he addressed the notion of
positive thinking — and shared with Fox News Digital what it would "look like to change your thinking and shift in a new direction."\n\nWASHINGTON PASTOR, ONCE
GRAVELY ILL, SHARES HOW ‘THE POWER OF THE SECOND THOUGHT’ TRANSFORMED HIS LIFE\n\nSaid Pastor Bradley, "Psychologists report that we have in between 6,000 and
60,000 thoughts a day, based on the person and situation. The National Science Foundation shares that up to 80% of our first thoughts are not helpful, even
destructive."\n\nWhat happens is that "when life is difficult, we can become flooded with thoughts that are drenched in fear, worry, stress, selfishness,
impurity, selfishness and lies.', 'Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle,
2023-10-28T09:00:12+00:00)\nWe do not have to believe, harbor or entertain negative first thoughts. We can be proactive by intentionally choosing our second
thought."\n\nIt is no accident that the pastor\'s new book, his second, is called "The Power of the Second Thought."\n\nIt dives into the power of understanding
that the first thoughts we have are not always the best — and that it\'s wise to take some time and get to the "second thoughts," which can yield greater
understanding.\n\nSTORYTELLER REVEALS HIS FAITH AT WORK: ‘I REFUSE TO TRY TO SCREW SOMEONE IN THE NAME OF BUSINESS’\n\nSaid Bradley to Fox News Digital in
emailed comments this week, "How can we choose a positive attitude and approach during the day? Reject and replace!"', 'Faith dilemma: How effective is positive
thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)\nReject and replace!"\n\nHe gave this list of
examples:\n\nReplace "People just slack and gossip at my job" with "I will work with all my strength, even when no one is looking"\n\nReplace "There\'s no hope
for my future" with "I am thankful for today and will keep doing the next right thing"\n\nReplace "My children are annoying" with "My children are a gift from
God"\n\nReplace "I\'ll never have good friendships" with "I will pursue and support the people that I respect the most"\n\nReplace "No one really cares about
me" with "God is with me and loves me every day"\n\nSaid Bradley, "Johns Hopkins reports that there are links between positive thinking and health. People with
a positive outlook are less likely to have a heart attack or cardiovascular event."\n\n"I didn\'t have a roadmap for finding hope or the tools to handle the
intensity of my loss and pain."\n\nAlso, "Research from the University of North Carolina points to positive people being more resilient and finding more
solutions to problems.', 'Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle,
2023-10-28T09:00:12+00:00)\nA Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your
opportunities to overall life satisfaction, positivity makes a difference."\n\nWASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH:
‘ASK GOD FOR COURAGE’\n\nThe pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental
health.\n\nBradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from.
He said finding hope during that time was challenging but is how he changes his "hopeless first thoughts."\n\n"I didn\'t have a roadmap for finding hope or the
tools to handle the intensity of my loss and pain," he said.\n\nFAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN\'T GONE TO CHURCH — WHAT DO I DO?\'', 'Faith
dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)\nHe said turning to
Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.\n\n"The Word of God is living and
active, bringing strength, courage, wisdom, and protection," he said.\n\n"God\'s thoughts and ways are higher than ours, but when we spend time, each day
reading His Word and listening to the voice of the Good Shepherd, we are filled with new thoughts that change the trajectory of our lives."\n\nBradley also
shared his dos and don\'ts of positive thinking.\n\nHe said it\'s not "denial and pretending that there are no challenges," he said.\n\n"Positive thinking is
not flattery and frivolous compliments."\n\nCLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER\n\nRather, positive thinking is realistic, full of truth and love
— and aligned with heaven and scripture.\n\n"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating
unrealistic expectations for other people and putting extra stress on them to be perfect."']
connections: [{'entity': 'Harvard', 'relationship': 'MENTIONS'}, {'entity': 'mental health', 'relationship': 'MENTIONS'}, {'entity': 'National Science
Foundation', 'relationship': 'MENTIONS'}, {'entity': 'health', 'relationship': 'MENTIONS'}, {'entity': 'Church', 'relationship': 'MENTIONS'}, {'entity': 'Jesse
Bradley', 'relationship': 'MENTIONS'}, {'entity': 'Grace Community Church', 'relationship': 'MENTIONS'}, {'entity': 'Auburn, Washington', 'relationship':
'MENTIONS'}, {'entity': 'faith', 'relationship': 'MENTIONS'}, {'entity': 'FOX News app', 'relationship': 'MENTIONS'}]
</item 6>

<item 7>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
A Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your opportunities to overall life
satisfaction, positivity makes a difference."

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

The pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental health.

Bradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from. He said
finding hope during that time was challenging but is how he changes his "hopeless first thoughts."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain," he said.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'

Graph facts:
- Want to help your kids achieve their potential? This expert has some advice -MENTIONS-> Harvard
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Harvard
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> mental health
- There’s something going on with AI startups in France -MENTIONS-> mental health
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> mental health

Entities: Harvard, mental health, Fox News, Church, loss and pain, Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in,
Dartmouth College, The Power of the Second Thought, optimism, Christian faith, hope, trauma, professional soccer, life satisfaction, Bradley
</item 7>

<item 8>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Reject and replace!"

He gave this list of examples:

Replace "People just slack and gossip at my job" with "I will work with all my strength, even when no one is looking"

Replace "There's no hope for my future" with "I am thankful for today and will keep doing the next right thing"

Replace "My children are annoying" with "My children are a gift from God"

Replace "I'll never have good friendships" with "I will pursue and support the people that I respect the most"

Replace "No one really cares about me" with "God is with me and loves me every day"

Said Bradley, "Johns Hopkins reports that there are links between positive thinking and health. People with a positive outlook are less likely to have a heart
attack or cardiovascular event."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain."

Also, "Research from the University of North Carolina points to positive people being more resilient and finding more solutions to problems.

Graph facts:
- Strategizing for sustainable development -MENTIONS-> health
- Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' -MENTIONS-> health
- What about the Mediterranean lifestyle reduces mortality? Scientists find out -MENTIONS-> health
- Israel thwarts Hamas on the financial battlefield -MENTIONS-> health
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -MENTIONS-> health

Entities: health, FOX News - Lifestyle, positive thinking, Johns Hopkins, University of North Carolina, heart attack, cardiovascular event, loss and pain,
resilience, finding solutions to problems, Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in, hope, Bradley, God
</item 8>

<item 9>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
CLICK HERE TO GET THE FOX NEWS APP

So it's challenged me on that level.

But as it challenges me, it also allows me to go further in that relationship — and to deepen my prayer life.

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -PUBLISHED_BY-> Fox News
- Ozempic, Wegovy may be linked to stomach paralysis and other digestive issues in large-scale study -PUBLISHED_BY-> Fox News
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -PUBLISHED_BY-> Fox News
- Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' -PUBLISHED_BY-> Fox News
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -PUBLISHED_BY-> Fox News

Entities: Fox News, faith, FOX News app, God, Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us', prayer life, Advent,
Jonathan Roumie
</item 9>

<item 10>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
HALLOW APP ANNOUNCES COLLABORATION WITH LIAM NEESON FOR NEW ADVENT SERIES THIS YEAR

People can "disconnect from the chaos of the real world, the divisiveness, the anxiety" that exists, said Roumie, by connecting with Jesus through prayer.

They can find "peace and a deeper relationship with Christ, and I think that's really the goal of the prayer challenge," he said.

Roumie said that building a habit of prayer "is easier than people think."

He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically. And as a result, you
then experience much more peace, tranquility and serenity in your life and the ability to really muscle through some of the most difficult obstacles and
challenges that life can throw at you," he said.

Graph facts:
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Liam Neeson
- Liam Neeson -SAID-> raising his sons on his own
- Liam Neeson -SAID-> grief
- Liam Neeson -SAID-> drinking too much
- Liam Neeson -PARTICIPATED_IN-> 2014 interview with 60 Minutes

Entities: Liam Neeson, FOX News - Lifestyle, Jesus, Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us', Advent, God,
prayer challenge, new Advent series, Jonathan Roumie, Hallow app, prayer
</item 10>
=== END ITEM id=32de29eb1810f1ad ===

=== ITEM id=c38bdc48963b62d6 ===
Gold sentences and facts:
Sentence 0: Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve.
  Fact 0.0: Inflation is fading.
  Fact 0.1: The fading of inflation leads the Fed to cut interest rates.
  Fact 0.2: Eventually, housing affordability will improve.
Sentence 1: While some economic factors are stronger today than they were during the GFC, housing affordability is undeniably worse.
  Fact 1.0: Some economic factors are stronger today than they were during the GFC.
  Fact 1.1: Housing affordability is undeniably worse today.

Retrieved items:
<item 1>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
29. “These headwinds are causing both buyers and sellers to hold out for better circumstances.”

But with the development that home sales transactions are at their lowest level in the past 13 years coupled with century-high mortgage rates of 8%, many real
estate experts and economists aren’t hopeful that affordability constraints will let up soon.

Out of fear of losing their lower interest rates, current homeowners are resistant to putting their homes on the market, ultimately leading to the abysmal
inventory levels—leaving fewer and fewer properties exchanging hands. Indeed, more than 90% of existing-home owners are locked into mortgage rates below 6%,
Odeta Kushi, deputy chief economist at First American, previously told Fortune.

“These homeowners do not have a financial incentive to sell,” Kushi says. “The combination of reduced affordability and an even stronger rate lock-in effect
suppresses home sales because you can’t buy what’s not for sale, even if you can afford it.”

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Inventory levels, Great Recession, The housing winter arrives early as existing home sales crash to Great Recession levels and the
long-feared ‘deep freeze’ sets in, First American, mortgage rates, Odeta Kushi, home sales, affordability, rate lock-in effect
</item 1>

<item 2>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Housing market pessimists have been sounding the alarm for years about a pending crash in the U.S. residential real estate market. Even before the Federal
Reserve began hiking interest rates to fight inflation last year, pushing mortgage rates to a 23-year high this month, years of surging home prices left some
experts warning that the housing market was a massive bubble ready to pop.

However, despite a widely bearish outlook in the industry, most real estate veterans have avoided arguing that home prices will drop like they did during the
2008 crash that kicked off the Global Financial Crisis (GFC). And so far, that’s been a wise decision.

It took until June 2022 for national home prices to peak, even as mortgage rates soared and purchase applications plummeted amid the Fed’s first interest rate
hikes. And while prices fell more than 5% from that peak by January of this year, they returned to a record high in July.

Graph facts:
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Federal Reserve
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Federal Reserve
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Federal Reserve
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Federal Reserve
- ASX to rise after interest rate hopes boost US shares -MENTIONS-> Federal Reserve

Entities: Federal Reserve, Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style, BofA, housing recession,
1980s style, 2008 crash, Global Financial Crisis, national home prices peak, home prices return to a record high, residential real estate market, housing
market, Fortune, interest rates, home prices, mortgage rates, inflation
</item 2>

<item 3>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
This can help the housing market retain some of its momentum without falling apart.”

Bank of America economists believe that low inventory of existing homes for sale combined with solid home sales to millennials could help put a “floor beneath
home prices,” but that doesn’t mean there won’t be some near-term pain due to higher mortgage rates.

“With rates likely staying higher for longer, we are cautious of potential turbulence ahead,” the group warned. Bank of America has previously forecast 0% home
price growth and falling home sales for the full year 2023, but it didn’t offer a new prediction in its latest note.

Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve. “At that stage, we should see a more stable and
healthy housing market,” Park and her team wrote. “Until then, hang tight, it may be a bumpy ride.”

This article has been updated to correct a mistaken term used to refer to Jeseo Park.
</item 3>

<item 4>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And while we haven’t seen home sales activity levels this low since the 2000s, the housing market more closely mimics that of the ’80s, according to a report
published this week by Fortune 500 financial services company First American.

“Today’s housing market isn’t anything like the housing market of the mid-2000s,” First American’s Fleming wrote in a Tuesday report titled “1980s Déjà Vu for
the Housing Market.” “The housing market today is not overbuilt, nor is it driven by loose lending standards, subprime mortgages, or homeowners who are highly
leveraged.”

While some economic factors are stronger today than they were during the GFC, housing affordability is undeniably worse.

“While housing and more generally consumer fundamentals are in a much stronger position today, affordability for the incremental buyer is worse than it was at
the peak in 2006 before the crash,” Roger Ashworth, a managing director at Goldman Sachs, wrote in a credit strategy research paper released last week.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Goldman Sachs
- OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO -MENTIONS-> Goldman Sachs
- Japan’s prime minister Kishida says: ‘reassess our economy’ -MENTIONS-> Goldman Sachs
- Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs -MENTIONS-> Goldman Sachs
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Goldman Sachs

Entities: Goldman Sachs, housing market, Fortune, Great Recession, The housing winter arrives early as existing home sales crash to Great Recession levels and
the long-feared ‘deep freeze’ sets in, First American, Fleming, housing affordability, GFC, Roger Ashworth, 1980s Déjà Vu for the Housing Market, loose lending
standards, subprime mortgages, consumer fundamentals, credit strategy research paper
</item 4>

<item 5>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Indeed, existing-home sales prices topped $306,000, a 5% increase since the year began, according to the Case-Shiller U.S. National Home Price Index.

“As has been the case throughout this year, limited inventory and low housing affordability continue to hamper home sales,” NAR chief economist Lawrence Yun
said in a statement. “The Federal Reserve simply cannot keep raising interest rates in light of softening inflation and weakening job gains.” The Fed has raised
rates several times this year, with Chair Jerome Powell arming his “higher for longer” stance.

Declining housing inventory levels also contribute to the drop in housing transactions. Housing inventory is down 8.1% from this time last year, according to
NAR. Some real-estate experts and economists argue that housing affordability is even worse now than during 2008, when a damaging downturn in the U.S.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Federal Reserve
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Federal Reserve
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Federal Reserve
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Federal Reserve
- ASX set for bright start despite Wall Street slip; $A drops -MENTIONS-> Federal Reserve

Entities: Federal Reserve, Fortune, interest rates, Existing-home sales, Great Recession, NAR, Lawrence Yun, Case-Shiller U.S. National Home Price Index, job
gains, The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in, housing, housing
affordability, housing inventory, inflation, Jerome Powell
</item 5>

<item 6>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.

Graph facts:
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> housing market
- 1980s Déjà Vu for the Housing Market -MENTIONS-> housing market
- Bank of America -SAID-> housing market
- Park -SAID-> housing market
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Fed

Entities: housing market, Fed, Fortune, interest rates, global economy, America, The housing winter arrives early as existing home sales crash to Great
Recession levels and the long-feared ‘deep freeze’ sets in, existing home sales crash, Great Recession, Great Financial Crisis, everything bubble, housing, rate
hikes, Zillow, deep freeze
</item 6>

<item 7>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Park and her team noted that there are no “noticeable signs” of excess housing development today like there were back then, and households aren’t nearly as
burdened by mortgage debt. Household mortgage debt represented roughly 65% of U.S. consumers’ disposable income in the second quarter, compared to 100% before
the GFC. And the ratio of Americans’ mortgage debt to their real estate assets—also called loan-to-value—was just 27% in the second quarter, compared to over
40% in 2008 and roughly 50% in 2010, Bank of America’s data shows.

New legislation was also enacted since the GFC to help prevent worst-case scenarios in the housing market. One of the most obvious effects of these new laws is
that there are far fewer risky adjustable-rate mortgages today.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-MENTIONS-> Bank of America
- Japan’s prime minister Kishida says: ‘reassess our economy’ -MENTIONS-> Bank of America
- From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner -MENTIONS-> Bank of America
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Bank of America
- ASX to rise after interest rate hopes boost US shares -MENTIONS-> Bank of America

Entities: Bank of America, Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style, excess housing development,
mortgage debt, loan-to-value, adjustable-rate mortgages, new legislation, housing market, housing recession, Park, Fortune, GFC, U.S.
</item 7>

<item 8>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Plus, the raising of interest rates in an effort to combat inflation is strikingly similar to the ’80s. Back then, Fed Chair Paul Volcker fought inflation
through aggressive interest rate hikes, with the average 30-year fixed mortgage peaking at about 18% by late 1981. Sound familiar? Current Fed Chair Jerome
Powell has set the tone for the most aggressive global hiking of rates in the modern era. And in due course, this week, mortgage rates hit 8%, the highest
they’ve been in more than 20 years.

All things considered, home sales activity also plummeted from 1978 to 1982. Existing-home sales dropped 50% during that period, according to the Office of
Policy Development and Research.

Graph facts:
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Paul Volcker
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Fed
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> Fed
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> Fed
- ASX rises after Wall Street gains on rate optimism; Oil drops -MENTIONS-> Fed

Entities: Paul Volcker, Fed, Fortune, interest rates, The housing winter arrives early as existing home sales crash to Great Recession levels and the
long-feared ‘deep freeze’ sets in, Great Recession, Office of Policy Development and Research, Existing-home sales dropped 50%, 30-year fixed mortgage peaked at
about 18%, Mortgage rates hit 8%, mortgage rates, home sales, inflation, Jerome Powell
</item 8>

<item 9>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The average 30-year fixed mortgage rate, the most common type of mortgage in the U.S., has soared from 3.8% in March 2022 to over 7.5% today as a result. This,
in turn, has slowed mortgage purchase applications and caused home sales to plummet, just as it did in the ’80s; but home prices, echoing the dynamic of that
era, have yet to collapse.

One of the key reasons for the resilience of the housing market in both of these periods is demographics. “Noticeably, demographics were favorable back then,
with baby boomers having entered the prime homebuying age,” Park and her team wrote Thursday, arguing millennials are in a similar position today. “Some sales
activity should be supported by millennials reaching the prime homebuying age, and single-family building permits have steadily held up. This can help the
housing market retain some of its momentum without falling apart.”

Graph facts:
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> 2008 crash
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Bank of America
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Federal Reserve
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> housing recession, 1980s style
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -MENTIONS-> Global Financial Crisis

Entities: Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style, BofA, housing market, housing recession, Park,
30-year fixed mortgage rate, mortgage purchase applications, demographics, baby boomers, millennials, single-family building permits, Fortune, home prices, home
sales, U.S.
</item 9>

<item 10>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And he’s not hopeful we’ll see home prices drop anytime soon: “Absent any negative shocks to the broader economy that would either boost excess supply of homes
on the market or fuel an uptick in unemployment, we continue to expect home prices to rise at a slow pace.” In fact, he predicts we’ll see home prices rise by
1.8% by the end of the year, with a 3.5% increase by the end of 2024.

Looking at more recent housing inventory data makes the issue even more stark. Between September 2018 and September 2023, the average number of homes on the
market dropped a whopping 60% to fewer than 700,000 active listings, according to Realtor.com.

“Unlike the turn of the millennium, house prices today are rising alongside mortgage rates, primarily due to low inventory,” Sam Khater, Freddie Mac’s chief
economist, said in a statement released Sept. 29.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in, Great
Recession, Realtor.com, Sam Khater, Freddie Mac, housing inventory, home prices, mortgage rates, unemployment, broader economy
</item 10>
=== END ITEM id=c38bdc48963b62d6 ===
