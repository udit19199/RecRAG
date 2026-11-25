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

This file has 2 items.

=== ITEM id=68967dd2fddc2e30 ===
Gold sentences and facts:
Sentence 0: The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive thinking.
  Fact 0.0: The pastor explained that Jesus brought more positivity than anyone.
  Fact 0.1: The pastor said it is important to keep faith aligned with the practice of positive thinking.
Sentence 1: He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically.
  Fact 1.0: The pastor added that when a person consistently taps into the habit of positive thinking, their relationship with Jesus grows deeper and deeper.
  Fact 1.1: The pastor said the growth in that relationship happens automatically.

Retrieved items:
<item 1>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
The scientists found that adoption of the Mediterranean lifestyle was indeed feasible and could be adapted to local contexts, such as availability of
ingredients, their affordability, local culinary traditions, and food preferences. Participants were found to have lowered their risk of cancer and
cardiovascular disease relative to the general population.

“We found that the three blocks of the MEDLIFE were independently associated with lower […] mortality,” explained Sotos-Prieto. “And within block 3 [physical
activity, rest, social habits, and conviviality] we found that participating in collective sports, limiting sedentary activities, and having adequate hours of
sleep (6-8 hours) were associated with lower mortality.”

However, there were some variables that could not be taken into account due to a lack of information. This was the case for olive oil consumption, a prominent
component of the Mediterranean diet previously associated with protective roles in many chronic diseases. “It is probable that we are underestimating the
Mediterranean lifestyle association,” said Sotos-Prieto.

To nap or not to nap?
</item 1>

<item 2>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
Through the early aughts, The Epoch Times grew from an online effort to a weekly physical newspaper, with a home base in New York and a TV production company,
New Tang Dynasty Television. It raised money from followers and was staffed by unpaid volunteers. It ran aggregated articles on international issues from Voice
of America next to Thanksgiving Day explainers, dispatches from Falun Gong parades, and exposés on atrocities alleged to have been committed by the Chinese
Communist Party.

By 2019, it had gone mostly digital and was spending millions of dollars on creating a network of Facebook pages and groups and running aggressive pro-Trump ad
campaigns. The move toward explicit support of Republicans, despite Li’s teachings to stay away from U.S. politics, was foreshadowed by Li’s comments at a Falun
Gong conference a year before.

Li said that Falun Gong’s media ought to put a “constructive” spin on the news, to advance the group’s aims.
</item 2>

<item 3>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“One way to start could be to emphasize the use of fresh, local products and ingredients in Mediterranean-style meals,” said Sotos-Prieto. “[But] changing
dietary habits and lifestyle is a gradual process, and affordability is also an important question.”

Other questions remain as to how governments should invest in introducing new healthy habits, which ones should be given preference, and how much change is it
enough to start seeing benefits.

The task is not simple since culture can define anything from ingredients and recipes to different social habits and work-life schedules. But Sotos-Prieto is
optimistic.

“I think that picking realistic goals, even if small, and sustaining them over time can make a difference [in public health],” concluded Sotos-Prieto.

Reference: Maroto-Rodriguez J. et al., Association of a Mediterranean Lifestyle With All-Cause and Cause-Specific Mortality: A Prospective Study from the UK
Biobank, Mayo Clinic proceedings (2023).
</item 3>

<item 4>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
Roumie: I've had so many people say they've cultivated a relationship with Jesus through prayer because they heard me talk about the app, or because they
watched "The Chosen" and then they started seeing this pop up in other places.

WHAT IS AN ADVENT AND WHAT DO ITS CANDLES MEAN DURING THIS HOLY SEASON?

And they've gotten introduced to the app and to a more deeply cultivated prayer life — and in some cases it's been absolutely transformative for people.

Some people who've never been to church started going to church. Some people who haven't been to church for 20 years have come back to the church. People have
converted — and all because they've fallen in love with the richness, the depth and the tradition of the Catholic Church.

And obviously [the Hallow app] is authentically Catholic, but it is for all Christians — Christians of all denominations. There's something there for everybody,
which is what makes the app so universally appealing.
</item 4>

<item 5>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
There's something there for everybody, which is what makes the app so universally appealing.

‘Simple act of surrender’

Question: You've described an experience earlier in your life in which you prayed about leaving all of your anxieties and worries aside, and instead trusted
that God would take care of things — and then after that came the role of Jesus in "The Chosen" for you. Is there a larger message here for people, things that
others might learn from your experience?

Roumie: I think it all comes down to the simple act of surrender to God in a person's life. And I went through this in a very visceral, substantial and
transformative way in my life five years ago.

I literally was on my knees and I said, "Jesus, I surrender myself to you. Take care of everything." And He did.
</item 5>

<item 6>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
It will also promote a green and blue economy coupled with improved governance to guarantee long-term climate and disaster resilience.

BUILDING RESILIENCE THROUGH SUSTAINABLE DEVELOPMENT

Coming off from the pandemic and with the current global economic climate, it is opportune for the government to proactively drive for actionable policies and
programs that focus on building the resiliency of the economy through sustainable development with greater emphasis given to addressing climate change.

Navigating external and domestic economic headwinds will not be an easy feat for the administration, but the private sector will be an important catalyst for
sustainable development. With the private sector sharing industry knowledge, resources and potentially even leading certain socio-economic programs and projects
of the National Government, it is to be hoped that AmBisyon Natin 2040 of long and healthy lives for Filipinos that are strongly rooted, comfortable, and secure
will be achieved.

This article is for general information only and is not a substitute for professional advice where the facts and circumstances warrant. The views and opinions
expressed above are those of the author and do not necessarily represent the views of SGV & Co.
</item 6>

<item 7>
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
</item 7>

<item 8>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Conversely, the switch from winter to summer time was found to have no effect.

People frequently report an increase in SAD and depressive symptoms in winter (Getty/iStock)

“This study shows that the transition from summer time to standard time was associated with an increase in the incidence rate of unipolar depressive episodes,”
the study authors concluded. “Distress associated with the sudden advancement of sunset, marking the coming of a long period of short days, may explain this
finding.”

A drop in heart attacks

Not everyone is convinced, however. Charlie Morley, author of Wake Up to Sleep, cites a 2014 study by the College of Cardiology, which reported a significant
drop in heart attacks in the 24 hours immediately following turning back the clocks.

“Fascinatingly, each year when the clocks go back for Daylight Saving and 1.6 billion people across 70 countries gain an extra hour in bed, there is a 21 per
cent decrease in heart attacks the next day,” he says.
</item 8>

<item 9>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Erin Palinski-Wade, a New Jersey-based dietitian with a focus on diabetes and nutrition, is skeptical that foods can be as addictive as drugs. (She was not
involved in the study.)

"Although foods rich in added sugar may stimulate the feel-good chemicals in the brain and become habit-forming, sugar itself is not addictive in the way
cocaine or another drug may be," she told Fox News Digital.

"Consuming sugar and then reducing or eliminating it from the diet will not result in withdrawal symptoms or side effects as would happen from a true
addiction," she continued.

"Food cravings are complex and tied into not just the nutrition profile of a food, but also the emotions and learned behaviors around eating."

APPLE A DAY — AND 4 OTHER GREAT FALL FOOD CHOICES THAT ARE HEALTHY, DELICIOUS AND FUN

Emotions, stress, overall metabolic health, sleep habits and learned feeding behaviors all shape people’s food choices, behaviors and responses, Palinski-Wade
said.
</item 9>

<item 10>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"One hundred percent of patients enrolled in Caring Contacts are receiving follow-up," noted Schalley.

"The program normalizes the struggles and challenges of mental health and stresses the importance of ongoing treatment."

Schalley credits the success of Caring Contacts to a multitude of factors, she told Fox News Digital.

"The program normalizes the struggles and challenges of mental health and stresses the importance of ongoing treatment," she said. "It provides positive coping
skills and words of encouragement."

Plus, the program "provides the ability for the youth to maintain a positive, trusting relationship with the social worker," said Schalley.

TEEN GIRLS DEALING WITH MENTAL HEALTH CRISIS NEED ‘CONNECTION’ MOST, DOCTORS SAY

Unexpected beneficiaries of the Caring Contacts program are the staff members who write the cards.

Participating in Caring Contacts has "provided [staff] with an avenue to receive responses from the youth regarding updates on how they have progressed and hear
feedback on their experience from the initial encounter," said Schalley.
</item 10>

<item 11>
How the conspiracy-fueled Epoch Times went mainstream and made millions (Hacker News, 2023-10-16T03:41:24+00:00)
With the new bedfellows came a new revenue stream. Though Falun Gong practitioners had been a reliable source of small donations in previous years, in 2020 the
group started to receive gifts and grants from big money conservative donors.

While The Epoch Times is not required to list its donors, it reported $8.4 million in revenue from contributions and grants in 2020 and 2021. Tax documents from
that period filed by scores of donors, accessed through ProPublica’s nonprofit explorer, show some of those funds came from conservative donors and foundations.

In 2021, The Epoch Times received $55,750 from the National Christian Charitable Foundation, which connects anonymous donors with Christian causes, and $31,000
from Donors Trust, a fund for conservative and libertarian donors. Smaller donations came from individuals — real estate agents, investors and surgeons among
them — and small family foundations, most of which support right-wing causes, including evangelical Christian groups, anti-vaccine groups and far-right media
organizations.
</item 11>

<item 12>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
Here is more of Fox News Digital's conversation with Roumie about the 25-day prayer challenge during Advent, which started on Monday — and about Roumie's own
faith beliefs.

‘Deep gratitude’

Fox News Digital: In the midst of some of the most trying times today, how do you — and how does anyone — keep a focus on prayer, on God, on a connection with
God? It's very, very easy to get distracted in today's world.

Jonathan Roumie: Yes, and I think that for me, I'm very easily distracted — and technology is one of the things that contributes to that distraction.

But in this case, I have such a deep gratitude to the technology that's arrived in the form of this app — of the Hallow app. I can literally go to my phone and
if I find myself or my mind wandering — or my anxiety building — all I have to do is literally click on the app and just start praying or meditating on one of
the challenges or one of the prayers.
</item 12>

<item 13>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
Take care of everything." And He did.

"It's a matter of tapping into [our] faith and allowing Him to be more involved in our lives — and surrendering is the first step to getting there"

And then three months later, I got called to do "The Chosen." It was a point that I think my life was building to when that happened — and I realized that
trying to micromanage God was not something that was working in my favor.

'HALLOW' PRAYER APP ACCOMPLISHED ITS GOAL DURING THE GOP DEBATE, COMPANY SAYS: 'PRAY WITH ME REAL QUICK'

And once I let him take the wheel, everything changed. The decisive moment was that simple prayer of surrender.

And I think that's really difficult for people, especially when we feel so out of control. When we turn on the news and see what's going on today, it feels like
we have no control in our own lives.

But God hasn't forsaken us.
</item 13>

<item 14>
Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages (FOX News - Entertainment,
2023-11-27T11:00:44+00:00)
The rule that works for them? "It’s always family first."

CLICK HERE TO SIGN UP FOR THE ENTERTAINMENT NEWSLETTER

WATCH: KEITH URBAN TALKS OPENING THE 2023 ACM AWARDS AND HIS MARRIAGE WITH NICOLE KIDMAN

"It's balanced, so it means it goes out of balance sometimes, and we just put it back in balance," Urban told Fox News Digital ahead of the 2023 ACM Awards this
year. "It's never perfectly in balance, but we get it back on track."

Kidman and Urban, both Australian, met in 2005. Kidman had previously been married to actor Tom Cruise, with whom she adopted two children over the course of
their marriage.

The lovebirds are parents to Sunday Rose and Faith Margaret.
</item 14>

<item 15>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Bradley said, "Positive thinking is relational, and there is no greater source of joy in our lives than God."
</item 15>

<item 16>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
It’s been proven that what people eat can help prevent or slow Alzheimer’s disease — but what about when they eat?

Participating in intermittent (time-restricted) fasting could lead to a reduced risk of cognitive deterioration, a recent study published in the journal Cell
Metabolism suggests.

Researchers at University of California San Diego School of Medicine adjusted the feeding schedule of certain groups of mice so that they only ate within
six-hour windows each day.

ALZHEIMER’S BLOOD TEST COULD HIT THE MARKET IN EARLY 2024, RESEARCHERS SAY

(This would be equivalent to 14 hours of fasting for humans, the researchers noted.)

Compared to a control group of mice that ate on demand, the fasting mice showed improvements in memory, were less hyperactive in the evenings and had fewer
sleep disruptions .

They also were shown to have less buildup of amyloid proteins in the brain, which is a hallmark of Alzheimer’s disease.
</item 16>

<item 17>
Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil (FOX News - Lifestyle, 2023-12-23T14:20:43+00:00)
AS THE JOY OF CHRISTMAS NEARS, FOUR FAITH LEADERS REVEAL ADVENT'S BLESSINGS: ‘WE LOOK FORWARD WITH HOPE'

On Christmas Day, 1863, the Civil War was a ravaging storm, ripping a gashing hole in the fabric of the Republic — while at the same time Longfellow suffered
from personal tragedies.

Fanny, his wife of 18 years and the mother of his six children, had died in a fiery accident in their home after her dress caught fire from a lit candle.
Longfellow's son Charlie was critically injured fighting for the Union army near Washington, D.C.

So on that day, the sound of the church bells heralding Christmas seemed to mock the very thought of "peace on Earth."

Yet through his personal grief and the torment of a war that would claim nearly a million lives, Longfellow penned his famous poem that has become a Christmas
carol for the ages, "I Heard the Bells on Christmas Day."
</item 17>

<item 18>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
ARTIFICIAL INTELLIGENCE HELPS DOCTORS PREDICT PATIENTS’ RISK OF DYING, STUDY FINDS: ‘SENSE OF URGENCY’

It’s important to protect data privacy and security as well, Patel noted, due to the large amounts of information that must be fed to AI models.

While AI's capabilities are "immense and ever-growing," Patel emphasized that AI’s role is to be an ally.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Humans caring for humans is the bedrock of health care — AI strengthens that foundation, but doesn’t replace it," he said.

"AI can offer insights and assist in decision-making, but human touch, intuition and empathy cannot be replicated by algorithms."

CLICK HERE TO GET THE FOX NEWS APP

While AI promises a "new frontier" in health care — Patel calls for a balance of "enthusiasm and caution."
</item 18>

<item 19>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
The motto traces its roots to the Old Testament, the foundational text of Western Civilization.

"In God have I put my trust; I will not be afraid what man can do unto me," states Psalm 56:11. A substantially similar phrase appears in the Quran.

However, trust in God became encoded in both America's foundational spirit and foundational documents.

"We are all equal before God. Religion does not make someone better than another person." — Pastor Bradley

"We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among
these are Life, Liberty and the pursuit of Happiness," reads the first principal listed in the Declaration of Independence.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'
</item 19>

<item 20>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 20>

<item 21>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
A Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your opportunities to overall life
satisfaction, positivity makes a difference."

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

The pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental health.

Bradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from. He said
finding hope during that time was challenging but is how he changes his "hopeless first thoughts."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain," he said.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'
</item 21>

<item 22>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Previous studies have supported this association between fasting and reduced Alzheimer’s risk.

Last year, a study led by the USC Leonard Davis School of Gerontology in Los Angeles also found that mice that followed a fasting diet showed lower levels of
"plaques and tangles" in the brain, reduced brain inflammation and better performance on cognitive tests.

"Cycles of a fasting-mimicking diet (FMD) had strong effects against Alzheimer’s symptoms in two different types of Alzheimer’s mouse models," the lead author
of that study, USC biogerontologist Dr. Valter Longo, told Fox News Digital.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain."

"In mice, the effects of FMD cycles were very strong and wide, meaning they affected both learning and memory and also Alzheimer’s pathology peptides/proteins
(amyloid and tau)."
</item 22>

<item 23>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Based on these findings, Longo recommends that people talk to their neurologist about considering fasting-mimicking diets and other nutrition-based approaches
in support of standard-of-care drugs and therapies.

"Especially for Alzheimer’s disease, for which pharmacological interventions have had very limited efficacy, it will be very important to continue to study
fasting-mimicking diets and other dietary interventions that can make drugs more effective," he added.

Desplats, lead author of the UC San Diego study, told Fox News Digital that Longo’s studies are on a similar track, demonstrating the connection between
metabolism, fasting and brain health.

"The main difference is that we tested an approach that does not require calorie reduction, which may be more sustainable and safer for older people whose
metabolisms are already impacted by aging changes," she said.

COLORADO WOMAN WITH ALZHEIMER’S IS STILL VIBRANT AFTER MORE THAN 4 YEARS: ‘I REFUSE TO SURRENDER’

Dr.
</item 23>

<item 24>
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
</item 24>

<item 25>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 25>

<item 26>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
In the last few years I’ve witnessed friends and family dashing to help parents with poorly designed online banking systems, unruly fridge alarms, “upgraded”
software that has changed beyond recognition, apps and phones that don’t function as expected, and numerous other technical challenges.

Frankly, technology is often baffling to me, too. Luckily I have kids who delight in parenting me – advising, explaining and guiding me on all things digital.
But studies suggest that this helpfulness might not be as beneficial as we think. Grappling with new and baffling things forces the ageing brain to work in
novel ways. A recent study from the University of Texas involving more than 200 elderly adults found that those who engaged with technology had better cognition
and memory than their non-engaging peers.

Digital engagement – whatever the device – protects the brain

Even the stress we experience while battling with our computer can be helpful, sharpening our mental acuity. The study authors concluded that digital engagement
– whatever the device – protects the brain.
</item 26>

<item 27>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
During his time in Belfast, Kalmus strummed a mandolin in a Celtic folk music group that meets at the eco-village, along with his son Zane, Wilcox-Hughes and
her daughter, Etta. He picked berries and ate fresh goat cheese and helped Ethan load hay into a neighbor’s barn.

climate scientist music group (Evan Bush / NBC News)

He spoke at the local library about climate change. He shared art, food and labor with neighbors. He saw how a community was sprouting up around the Possibility
Alliance — people willing to sacrifice the conveniences of a modern life built on fossil fuels.

He felt safer here and more energized to work.

It wasn’t about the place, but the people, he realized.

People at the center

Maine, like anywhere else, isn’t immune to climate change. But here, at least, he had people who understood his angst, who could share music and joy, who were
willing to make their own sacrifices, and who were trying to prepare their community in the way they thought was best.
</item 27>

<item 28>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
The study authors concluded that digital engagement – whatever the device – protects the brain. These findings reflect earlier studies that discovered that
older people who made active use of technology were less likely to get dementia.

Dr Eamon Laird, researcher in health and ageing at Limerick university, agrees that we should be encouraging older parents to try new things. And the further
out of their comfort zone they feel, the better. “It’s always good to keep the mind active and fresh,” he told me. “New challenges can help build and maintain
new brain connections and can be good for brain and overall health.”

If older people use technology to keep in touch with extended family and friends, that’s a double whammy. When my 84-year-old mother recently joined Instagram,
she not only expanded her grey matter but she also found an effective means of staying in touch with her grandchildren.
</item 28>

<item 29>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 29>

<item 30>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
In a phone interview with Fox News Digital on Monday, Los Angeles-based actor Jonathan Roumie spoke of his Christian faith, his prayer life and his hope that
others may find peace and serenity during this Advent season of 2023 — because "any opportunity to connect to Jesus is a welcome one," he said.

Roumie, perhaps best known for his role as Jesus Christ in "The Chosen," has paired up with Hallow, the No. 1 prayer app in the world, for a special Advent
prayer challenge for this holy season of preparation for Christmas. The 25-day prayer challenge began Monday — and as part of the effort, both he and actor Liam
Neeson guide people through a variety of prayers, meditations, Bible readings and more.

This is no mere gig, however, for Roumie.

He is a committed Christian and Catholic, with his own personal story of growing closer to God and continuing to do so each day.
</item 30>

<item 31>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
People will lose jobs. And action, alas, will be further delayed.

In the end, it’s simple: Inconsistency breeds uncertainty. So give us consistency. It is possible for our sector to succeed in spite of bad policies. But to
thrive, we need consistent resolutions and behavior from a government that actually cares about leading in this critical fight. Given the colossal stakes, which
were at least recently taken seriously, is that really asking too much?
</item 31>

<item 32>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
We do not have to believe, harbor or entertain negative first thoughts. We can be proactive by intentionally choosing our second thought."

It is no accident that the pastor's new book, his second, is called "The Power of the Second Thought."

It dives into the power of understanding that the first thoughts we have are not always the best — and that it's wise to take some time and get to the "second
thoughts," which can yield greater understanding.

STORYTELLER REVEALS HIS FAITH AT WORK: ‘I REFUSE TO TRY TO SCREW SOMEONE IN THE NAME OF BUSINESS’

Said Bradley to Fox News Digital in emailed comments this week, "How can we choose a positive attitude and approach during the day? Reject and replace!"
</item 32>

<item 33>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
HALLOW APP ANNOUNCES COLLABORATION WITH LIAM NEESON FOR NEW ADVENT SERIES THIS YEAR

People can "disconnect from the chaos of the real world, the divisiveness, the anxiety" that exists, said Roumie, by connecting with Jesus through prayer.

They can find "peace and a deeper relationship with Christ, and I think that's really the goal of the prayer challenge," he said.

Roumie said that building a habit of prayer "is easier than people think."

He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically. And as a result, you
then experience much more peace, tranquility and serenity in your life and the ability to really muscle through some of the most difficult obstacles and
challenges that life can throw at you," he said.
</item 33>

<item 34>
Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil (FOX News - Lifestyle, 2023-12-23T14:20:43+00:00)
But nothing really seems to change … A lot of that has to do with human nature."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Longfellow's poem still bears the truth because it was based on divine reality.

Yes, hate is strong — but God's love is stronger.

Wars do bring death — but God through Jesus Christ has conquered death.

And because God is now "with us," there is hope ... even with a broken heart.

CLICK HERE TO GET THE FOX NEWS APP

And there is triumph even amid tragedy.

This Christmas Day, strain your ears, push aside the din of the culture, the city, the war, the pain … and hear the bells.
</item 34>

<item 35>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
And if you want something that's very specific, it is here. Plus, there are resources for mental health.

There is music.

And for me as a musician, music is so transportive and nourishing for the spirit — to be able to listen to music that kind of gets you in the mindset for prayer
and worship.

There are just so many options that people have that it's kind of mind-bending.

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Question: How has playing the role of Jesus in "The Chosen" changed your life and changed you?

Roumie: It's allowed me to go deeper into my own relationship with Him.

It's caused me to examine what exactly is my relationship with Him — and how can I go further in that relationship, how can I get close to Him, how can I grow
deeper in communion with Him.

CLICK HERE TO GET THE FOX NEWS APP

So it's challenged me on that level.
</item 35>

<item 36>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
Despite the problems with social media and the urgency, in terms of its impact on society, Ohanian believes there’s still some hope. He noted, for example, that
the founders his fund 776 invests in are much smarter and more aware of the issues than he ever was at their age. Perhaps the next platforms will still be
addictive, but in a way that doesn’t “further disrupt democracy,” he mused.

He also believes that after everything on social media becomes so perfectly curated for us, our brains will start to crave things that are not. “I think we’ll
see a rebirth of live gatherings and in-person experiences…the bright side of this is that I hope it then gets us back in touch with our humanity, and the
things that no amount of AI can replace.”

“I want to be an optimist about it,” Ohanian added. “And I think there’s still a path forward but we’re fighting against very, very strong human things.
</item 36>

<item 37>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
The U.S. Congress voted to reaffirm "In God We Trust" as the national motto by a bote of 396-6 in 2011.

But faith in God today clearly faces challenges — with society at large likely suffering.

"I believe we are at a crossroads spiritually," said Pastor Bradley.

"God knows us completely and loves us perfectly. It’s good to have trust written on our currency, but what really matters is that it is written on our hearts."

CLICK HERE TO GET THE FOX NEWS APP

"We have wandered away from God in countless ways, including idolatry, materialism, pride and hatred."

He added, "May God have mercy on us in spite of our rebellion."
</item 37>

<item 38>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
Ouija boards and other attempts to converse with the dead are spiritually dangerous practices that should not be viewed as innocent fun, an exorcist priest told
Fox News Digital.

"We too often forget that there is much more to this world than meets the eye," Fr. Dan Reehil told Fox News Digital. "Angels, demons, spirits and souls do
exist and they can have an impact on our life."

Rehill is a priest of the Diocese of Nashville. In 2018, he received exorcism training at Ateneo Pontificio Regina Apostolorum in Rome, and was installed as the
exorcist of the Diocese of Nashville that same year.

DO ANGELS EXIST? OVER HALF OF AMERICANS SAY YES, STUDY FINDS: ‘COMFORT AND REASSURANCE’

An exorcist, said Reehil, "is a priest chosen and installed by a bishop to exercise the office of deliverance ministry.
</item 38>

<item 39>
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
</item 39>

<item 40>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 40>

<item 41>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
CLICK HERE TO GET THE FOX NEWS APP

So it's challenged me on that level.

But as it challenges me, it also allows me to go further in that relationship — and to deepen my prayer life.
</item 41>

<item 42>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 42>

<item 43>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Roosters crowed. A tear dripped down Ethan’s cheek.

“Let’s dedicate our work today to the Amazon,” Ethan said, outlining for the group how the work planned would sequester carbon and show people they could live
without fossil fuels and that they could take meaningful action against climate change, even as world governments failed them.

(Estimates of when the Amazon could “tip” and transition to savannah vary. Timothy Lenton, a University of Exeter professor who studies climate tipping points
and the Amazon’s resilience, said most estimates suggest the Amazon could cross a threshold as soon as 2 degrees C of warming.)

Life at the Possibility Alliance carried on. Will Foley, who hitchhiked from Florida to stay at the homestead, swung a scythe to clear brush. Frankie Williams
and Max Kurke, recent college graduates living at the alliance as summer apprentices, piled wood while wearing oversize sun hats. Ethan corralled goats for “mob
grazing,” a process he says helps sequester carbon.
</item 43>

<item 44>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
Faith in God in the United States was not just a statement of religious faith, but a declaration of national identity and human equality.

God, the Creator in this case, was non-denominational — which itself became essential the principal of religious freedom in the United States.

"The motto doesn't nullify the freedom to choose faith. Nothing is forced," said the pastor.

NEW YORK VETERAN, 105, SHARES HER SECRET TO A FULFILLGN LIFE: FAITH, FAMILY AND ‘NO REGRETS’

"We are all equal before God. Religion does not make someone better than another person."

The Founding Fathers paved the way for a society that enjoyed both faith in a Creator and religious freedom.

Trust in God, as the source of individual liberty and a defense against abuses of man, were bedrock beliefs among the Founding Fathers, regardless of individual
philosophies.
</item 44>

<item 45>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
In the past, Poyer’s was filled with posts of other girls, according to Rachel. Now there’s nothing on Poyer’s For You page except football and ayahuasca. She
knows because they have one another’s passwords.

“His attention now is 100 percent on me, his family and his career,” says Rachel, who has more than 4 million Instagram followers, an OnlyFans page and a line
of skincare products.

Some players feel more pressure when the end is closer than the beginning. The Bills have an out in Poyer’s contract after this season. He isn’t stressing about
any of it. His focus is here and now.

Poyer has thought about becoming a football commentator after retirement. But he has more to offer than sports talk. He could talk about why he’s never had a
vaccine, why Aliyah is homeschooled, or how he believes God transcends religions.

He could discuss why he has always thought he might be an alien.
</item 45>

<item 46>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
These three levels are temptation, obsession (which is also called "oppression") and possession.

'HALLOW' PRAYER APP ACCOMPLISHED ITS GOAL DURING THE GOP DEBATE, COMPANY SAYS: 'PRAY WITH ME REAL QUICK'

Temptation, said Reehil, is something experienced by everyone.

Obsession is "when demons torment people physically, spiritually, mentally or emotionally, or some combination of these."

Possession, which is arguably the most well-known of the three levels, is when a demon or demons "take physical possession of a person's body and make use of
its faculties," said Reehil.

"Possession is usually intermittent, almost never an uninterrupted possessing of the person’s body," he added.

Attempting occult practices, such as purportedly trying to contact the dead via an Ouija board, are a way to open oneself up to demonic activity inadvertently,
said Reehil.
</item 46>

<item 47>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
But he found himself daydreaming of a simpler life, where he could keep bees, grow vegetables and press cider to drink on a Friday night and live closer to the
land.

Visiting the Possibility Alliance, a sprawling 11-acre homestead filled with fruit trees, goats, chickens and a garden, allowed him to scratch an itch he’d
entertained for much of his adult life.

The Hughes family, which operates the homestead, and its guests live largely without electricity or modern technology.

The family of four doesn’t fly because of climate concerns and also doesn't own a car. Their main use of fossil fuels is to take passenger trains to climate
protests.

They try to eschew capitalism and have built, instead, a "gift economy" where resources are shared and skills are traded among neighbors in this tiny corner of
Maine. They grow much of their own food, hold trainings for climate protesters and plan to host refugees as climate disasters worsen.

Nights are illuminated by candlelight. Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.
</item 47>

<item 48>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
after newsletter promotion

This is what Ten Hag has overseen in recent weeks as the protracted sale of the club has entered a new phase, with Sir Jim Ratcliffe’s bid to buy 25% for an
inflated £1.3bn in return for control of football policy expected to go through imminently. But despite any implications this may have for the futures of Ten
Hag’s players, the manager himself does not think the matter is a distraction.

Bruno Fernandes volleys home for Manchester United against Burnley – one of two goals he has scored for the club this season. Photograph: Matthew
Peters/Manchester United/Getty Images

“Of course, they are committed to the club but especially they are committed to this team,” Ten Hag says. “They want to be successful and are therefore aware
you need [to focus on] the team, your teammates. That is how the players are thinking and they are not thinking about strategic reviews or structures or
whatever.”

On Sunday at 3.30pm we will find out just how focused United’s players are.
</item 48>

<item 49>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
It’s hard. This s— is hard. And it’s not just (the losses). I’m 32, not this young spring chicken anymore. You know, s— hurts all the time. But my perspective
is I get to play a game that I love, and I’m healthy enough to play. Every time you come out of a game, that’s a win in its own way.”

GO DEEPER Bills training camp preview: Is 2023 final run for Micah Hyde-Jordan Poyer tandem?

Poyer’s commitment is such that last season, when a rib injury precluded him from airplane travel, he took a 32-hour round-trip drive to play in a game in
Kansas City. He wants to lift a Lombardi Trophy badly, as much as ever. But if it doesn’t happen, he believes he can handle the disappointment better.

On Fridays this season, Poyer, fellow safety Taylor Rapp and assistant strength coach Will Greenberg play crystal bowls at Bills headquarters.
</item 49>

<item 50>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The researchers hope that these findings will lead to human trials.

The goal is to obtain funding to start a pilot study of patients next year.

"Accessible interventions like this have great translational potential, as they are available and are usually very well-tolerated," Desplats said.

The goal is to obtain funding to start a pilot study of patients next year.

"This may have special importance for the elderly , as time-restricted eating does not require limitation in calories or change in the diet, but may offer
important benefits ranging from metabolic and sleep regulation to potential cognitive improvement."

TIME-RESTRICTED EATING NO MORE BENEFICIAL THAN CALORIC RESTRICTION IN OBESE PATIENTS, STUDY SAYS

Adopting a new dietary pattern should not be a replacement for medical care , however, as Desplats emphasized the importance of patients' discussions about
options with their clinicians.
</item 50>

<item 51>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
after newsletter promotion

And clarity is perhaps what this team has been lacking most of late: clarity of selection, clarity of formation, clarity of mission and motivation. How much
prestige should we really attach to the Nations League, anyway? Is it primarily an Olympic qualification shootout or something worth winning in its own right?
England ended up trying to answer these questions on the hoof, and somehow it was only when their options had finally clarified down to one – survival or
oblivion – that they recovered their poise and verve.

This entire post-pandemic international cycle – a Euros, a World Cup, an Olympics and another Euros squeezed into three years – feels like a kind of sadism
dressed up as a panorama of opportunity. There is a school of thought out there that a fallow summer might not have been the worst thing for this team: time to
reflect and process, time to breathe, time to refresh.

Instead, they now travel to Glasgow on Tuesday for one last push. The minds are still frazzled.
</item 51>

<item 52>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
He could discuss why he has always thought he might be an alien. “I truly feel like I’m not from this planet, always being the odd one out, having different
perspectives, thinking differently,” says Poyer, who is fascinated by stars.

Poyer has what philosophers call epistemic humility — he believes knowledge is limited and filtered by personal experiences. And now he is beginning to solve
the mysteries of his existence.

Advertisement

“My life has been changed forever,” he says. “And my purpose is to be a bright light for everybody that I touch and connect with.”

In March, Poyer will return to Resonance for another retreat. Also planning to go are his cousin who went with him last year, and some first-timers — his
mother, brother, a couple of friends and possibly a couple of teammates. He will be working on a documentary about ayahuasca.

A handpan is a steel musical instrument that generates sound through vibrations. Sound healers and meditators often play the handpan.
</item 52>

<item 53>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
Rachel has no interest in taking ayahuasca. But she is grateful her husband discovered it.

Advertisement

“After he did ayahuasca, he became like my dream husband,” Rachel says. “I was nervous about what it would do to him, but he came back a totally new man, so
appreciative of me and our family.”

Bills safety Jordan Poyer with his wife, Rachel, and their daughter, Aliyah. (Courtesy of Rachel Bush Poyer)

Highest-version Poyer made Poyer aware he was not doing the little things — waking up with a smile, giving Rachel a hug on his way out, or being present even
when his attention could be divided.

Poyer has turned off his DMs on Instagram and deleted his Twitter/X account. Instagram features a For You page — a personalized feed based on user engagement
patterns. In the past, Poyer’s was filled with posts of other girls, according to Rachel.
</item 53>

<item 54>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Welcome to Startups Weekly. Sign up here to get it in your inbox every Friday.

A few months ago, Alexis Ohanian, the co-founder of Reddit, tweeted that if he could go back in time and do one thing differently when he was building Reddit,
he would have spent significantly less time attending events. Personally I have a different experience — I always regret getting on a plane (or in an Uber, for
that matter) to go to an event. However, as I’m traveling back, I’ve never regretted it: There’s always something magical that comes out of going to an event,
in my experience.

Our TC+ team is not interested in anecdotes, however: neither mine nor Alexis’. So, like the data-driven journalists they are, they surveyed more than 50
founders, trying to figure out whether attending events still makes sense in 2023.

Spoiler alert: There isn’t a real consensus, but there are some really good pros and cons.
</item 54>

<item 55>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.
</item 55>

<item 56>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Being social, says Laird, is as important as trying new things: “Research shows that those who are more social are also more resilient to chronic conditions and
often have a stronger immune system.”

Anyone with older parents should be urging them out for a walk five days a week Dr Eamon Laird, health and ageing researcher

Depression among the elderly is on the rise and often goes unnoticed. Gerontologists now believe that it is one of the most common psychiatric disorders among
the elderly – with half of all cases going undiagnosed. When elderly parents become depressed (many of them come from a generation not accustomed to talking
about their mental health), it can be difficult to know how to respond. But Laird’s latest report suggests that a daily walk could help. His study of more than
4,000 older adults found that as little as 20 minutes of daily walking could cut the risk of depression by up to 43%.
</item 56>

<item 57>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
“What I really wanted to do was work on trying to be a better me,” he says.

He discovered Resonance, a retreat center in Costa Rica on a hilltop overlooking the Pacific coast and the cloud forest of Monteverde.

To prepare his mind to enter a state of awareness, he was to eat only chicken, fish and salad for one month, with no oils and sugar added. Then he had to cut
out chicken and fish for the week preceding the ceremony. He also was told to practice yoga and breathing exercises that would be useful if he experienced
anxiety during the three ceremonies spread over a week.

In a tiki hut in a wooded, secluded area, a shaman presented each participant with a cup of ayahuasca brew, which is kind of like tea, but with benefits. Taste
is not one of them. Poyer says ayahuasca tastes like earth.

“It’s brownish and really thick,” he says. “It almost looks like oil, but it’s darker and thicker.
</item 57>

<item 58>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Scholars have argued that putting too much emphasis on the platforms can be too simplistic and reeks of technological determinism; they have also pointed out
that cultures like the United States’ and the legacy media have a long history with post-truth. That said, there are certainly dynamics going on that we did not
anticipate, and we don’t seem quite sure what to do with them, even with multiple areas of scholarship in communication, disinformation, and social media and
democracy working on these inquiries for years.

What seems clear is that the Internet is not the connectedness we imagined. The ecology and spirituality of the 1960s, which shaped and structured much of what
we see as energy democracy and the good future today, told us we were all connected. Globally networked — it sounds familiar, like a fevered dream from the
1980s or 1990s, a dream that in turn had its roots in the 1960s and before.
</item 58>

<item 59>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
Players reacted to last Friday’s news by sharing defiant messages on their WhatsApp group. After a run of six wins in nine games in all competitions, that had
lifted Everton to 14th in the table prior to being plunged back to 19th, their resolve to maintain momentum and haul the club out of danger was relayed to the
manager at Friday’s meeting.

“The feeling was very positive in the sense of building on the mentality that’s been there lately,” said Dyche. “We all spoke about that, saying this hasn’t
changed it and it’s building regardless of this current situation. The players are clear-minded. They have their WhatsApp group and when the news broke they put
it straight on there: ‘Right lads, just another challenge, this is how we take it on’. The group is taking ownership and that’s a powerful thing.

“I got from them the sense that: ‘What we’re building, we don’t let anything disrupt that and we don’t let anything get in the way’.
</item 59>

<item 60>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
"And this meant a lot of time looking at each other on-screen instead of in person."

Long after the pandemic, many of these habits have persisted.

"This video-ing of each other, it turns out, is far more tiring to humans than actually being together," said Nassar.

"One reason is that our brains are wired for actual in-person interactions, not wired for interactions on the screen."

Impacts of Zoom fatigue

While video calls can be convenient, Elliott warned that staring at a computer or phone screen for extended periods of time can result in negative mental health
consequences .

"Zoom fatigue can aggravate stress, anxiety and burnout among workers," he told Fox News Digital.

"Our brains are wired for actual in-person interactions, not wired for interactions on the screen."

"Some people share that spending so much time interacting on screens makes it difficult to interact in real life."

CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT?
</item 60>

<item 61>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
With this in mind, parents should avoid "processing their frustration around the holidays audibly around the children," she said.

"The ‘mom brain’ is wired to process problems rapidly and move on to solutions," she said. "The ‘child brain,’ however, is wired to process a problem internally
and take on the responsibility."

Said Keeley, "Knowing this, make sure you don’t do any ‘grinchy griping’ out loud to the children. You may see it as temporary venting, but they see it as, ‘Mom
is upset and I’m probably the reason.’"

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Instead, parents should stay calm, find a reason to be grateful and vocalize it, even when frustrations accumulate.

5. Neglecting the ‘give list’

While building a wish list is fun for kids, Keeley emphasized the moral importance of the "give list."
</item 61>

<item 62>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“Meals are something that happens around family and friends,” she continued. “The conviviality and social aspect are also characteristic. Physical activity is
important in the context of community life, and adequate rest including the typical short mid nap or siesta.”

Sotos-Prieto and researchers from the Universidad Autónoma de Madrid in Spain and the Harvard Chan School of Public Health in the USA are exploring the
feasibility and impact of adopting the Mediterranean diet and lifestyle among non-Mediterranean populations using locally available products and within their
own cultural contexts.

“We are seeing the transferability of the lifestyle [outside the Mediterranean] and its positive effects on health,” she added. The most significant perk? A 29%
lower risk of mortality in both men and women.

Breaking down the effects

Scientists stress that the Mediterranean diet should be understood more as a way of life than a mere dietary pattern with scientific evidence acquired over the
last four decades establishing its benefits on life expectancy, quality of life, and the prevention of several chronic diseases.
</item 62>

<item 63>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
Trust in God has been a foundational principle of the United States since the days of colonial settlement.

The phrase "In God We Trust" has been an official statement of national faith since the Civil War and the official motto of the United States since 1956.

It remains just as essential today as the United States faces cultural upheaval with numerous signs that faith is waning, Pastor Jesse Bradley of Grace
Community Church in Auburn, Washington, near Seattle, told Fox News Digital this week.

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

"Abiding with God goes beyond and is more valuable than money, technology and entertainment," said Pastor Bradley, in a series of thoughtful comments when asked
to tackle the state of America’s relationship with God.

"The soul of a nation is the core, the essence and the deepest part of a country," he said.
</item 63>

<item 64>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 64>

<item 65>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
(Sarah Stier / Getty Images)

Two days before OTAs began last April, Poyer returned to the Bills’ headquarters and stepped on the scale. The needle settled at 179 pounds, about 20 pounds
below his playing weight.

After resuming his regular routine, the weight came back quickly.

He discussed his experience with curious teammates, coaches and the team psychologist. Poyer hoped ayahuasca would elevate his football performances as Rodgers
believes it did his, but it hasn’t worked that way for him.

He is the second-leading tackler on his team but hasn’t made as many big plays as he would like. What he has done, according to former Bills defensive
coordinator Les Frazier, now an analyst for NFL Network, is enable his defense to be the best it can by lifting others through versatility, sacrifice, toughness
and communication. Poyer wears his “C” well.

“I feel like I’m playing OK this year,” Poyer said during the Bills’ bye week on the patio of his Florida home.
</item 65>

<item 66>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Unfortunately for anyone hoping for substantive federal regulation in the U.S., 2024 is not the year to expect movement on that front. Though it will be a year
for AI and everyone will be asking for new laws, the U.S. government and electorate will be too busy with the trash fire that will be the 2024 election.

The 2024 election is a trash fire and AI makes it worse

How the 2024 presidential election will play out is, really, anyone’s guess right now. Too many things are up in the air to make any real predictions except
that, as before, the influence mongers will use every tool in the box to move the needle, including AI in whatever form is convenient.

For instance, expect bot accounts and fake blogs to spout generated nonsense 24/7. A few people working full time with a text and image generator can cover a
lot of ground, generating hundreds of social media and blog posts with totally fabricated images and news.
</item 66>

<item 67>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He has been on a 15-year quest to halt climate change and has already oriented most everything in his life to minimize his own climate footprint and maximize
what he can do to push society to change.

Nothing has been enough.

And so this summer, as extreme heat and rainfall battered many American communities, Kalmus visited Maine searching for something different. He was looking not
for climate solutions, but for new ways to withstand climate change’s harms — and also to make peace with the grief that often feels inescapable for those who
reckon with global warming daily.

climate scientist exercise bicycle environmentalist (Evan Bush / NBC News)

That search brought him to an unusual community called the Possibility Alliance, a homestead where a small group of people is trying to figure out how to live
in a world headed for a breakdown (as they see it).

Its leaders eschew capitalism, grow much of their own food, and live without electricity. They’ve dedicated their lives to protest, living off the land as
simply as possible, and are preparing their community for ecological disaster and to host climate refugees.
</item 67>

<item 68>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Climate change isn't going away, and it isn't going to get any better — at least if we keep legislating as we have been. In Democracy in a Hotter Time: Climate
Change and Democratic Transformation, a multidisciplinary collection of subject matter experts discuss the increasingly intertwined fates of American ecology
and democracy, arguing that only by strengthening our existing institutions will we be able to weather the oncoming "long emergency."

In the excerpt below, contributing author and Assistant Professor of Environment and Sustainability at the University at Buffalo, Holly Jean Buck, explores how
accelerating climate change, the modern internet and authoritarianism's recent renaissance are influencing and amplifying one another's negative impacts, to the
detriment of us all.

MIT Press

Excerpted from Democracy in a Hotter Time: Climate Change and Democratic Transformation, edited by David W. Orr. Published by MIT Press. Copyright © 2023. All
rights reserved.
</item 68>

<item 69>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
"America needs to come to her senses and realize the goodness of God."

There are numerous signs that individual welfare and social cohesion suffer as God becomes a part-time or non-existent guide in our lives and our nation.

"We can't compartmentalize God into one day, hour or place," said Pastor Bradley. "This relationship with our Maker is 24-7, as God is the source of our
strength, love, security and hope."

MICHIGAN SIBLINGS REUNITE WITH THEIR 3-YEAR-OLD BROTHER AFTER HIS 6-MONTH HOSPITAL STAY FOR CANCER TREATMENT

"In God We Trust" is not a uniquely Judeo-Christian American concept. Some form of it has been expressed by other cultures, nations and faiths throughout
history.

The motto traces its roots to the Old Testament, the foundational text of Western Civilization.
</item 69>

<item 70>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Joseph Antoun, CEO and chairman of the nutritech company L-Nutra in California, was also involved in the USC study.

"In these studies, mice fed the fasting-mimicking diet showed improved cognitive performance, motor learning and both short- and long-term memory, and promoted
the growth of new neurons in the brain," he told Fox News Digital via email.

"Fasting-mimicking diets have the potential to drastically reduce many aging and disease risk factors by reducing neuroinflammation or superoxide production in
the brain."

Fasting can help the brain function in a myriad of ways, said Antoun.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain," he told Fox News Digital in an emailed
statement.

Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
</item 70>

<item 71>
Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil (FOX News - Lifestyle, 2023-12-23T14:20:43+00:00)
Read this article for free! Plus get unlimited access to thousands of articles, videos and more with your free account! Please enter a valid email address. By
entering your email, you are agreeing to Fox News Terms of Service and Privacy Policy , which includes our Notice of Financial Incentive . To access the
content, check your email and follow the instructions provided.

How can there be "Joy To the World" when there is so much sorrow on Earth?

How can there be thoughts of sleigh bells jingling, chestnuts roasting on an open fire or Jack Frost nipping at our noses when war rages in Israel and Ukraine,
and deep hatred of God's chosen people continues seething in our nation's streets?

Henry Wadsworth Longfellow, one of America's greatest poets, thought the same exactly 160 years ago.
</item 71>

<item 72>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 72>

<item 73>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 73>

<item 74>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
This type of chart illustrates situations in which low values of one variable (in this case, a short nap duration) lead to a decrease in the values of the
second variable (incidence of cardiovascular events), while the trend reverses at higher values.

“Short naps, which range from 0 to 30 minutes, were associated with lower incident cardiovascular disease,” said Sotos-Preito. “An inverse association between
short naps and mortality and cardiovascular diseases was found in our previous publications.”

When asked for a definitive answer as to whether napping was indeed beneficial or not, Sotos-Prieto indicated that more research is needed in this regard.

Adapting to the lifestyle

The challenge remains in how to put these results into practice in other countries with different, deep-rooted customs. Cultural adaptation of beneficial
practices is essential to make the Mediterranean diet and habits more appealing and practical for diverse populations in non-Mediterranean countries.
</item 74>

<item 75>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
This season, NFL’s Thanksgiving Day games created history. Across all the three games, the NFL recorded an average viewership of 34.1 million. These figures
surpassed last year’s record viewership of 33.6 million. Furthermore, the combined unduplicated audience for all three games reached 133 million, marking the
second-highest total audience on Thanksgiving, surpassed only by last year’s numbers.

These figures paint a pretty one-sided picture, but the choice is much more nuanced. Comparing the NBA Christmas Day Games and NFL Thanksgiving Day Games comes
down to personal preference. Each tradition holds its own significance, adding to the diverse fabric of American sports culture. Whether one prefers the slam
dunks and three-pointers of the NBA or the hard-hitting action on the football field, both traditions offer a festive and entertaining way to celebrate holidays
with family and friends.

Are the NFL’s new decisions threatening the NBA’s Christmas Day dominance?
</item 75>

<item 76>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Sabrina Schalley, director of care coordination at Children's Nebraska, a pediatric hospital and medical center in Omaha, Nebraska, told Fox News Digital that
Caring Contacts was inspired by a similar program for adult survivors of suicide.

In the fall of 2019, Schalley brought the idea of personalized follow-up care to the emergency department's social workers, "and they all liked the idea of the
Caring Contacts."

The program was set to begin on May 1, 2020 — and that timing would turn out to be rather providential.

"We had no idea when we started that COVID was around the corner and the impact that would have on children's mental health," said Schalley.

"We quickly became so grateful for the intervention."

TEEN GIRLS ARE STRUGGLING WITH MENTAL ILLNESS AT RECORD LEVELS, WITH MANY 'PERSISTENTLY SAD,' DATA REVEALS

In the years since the COVID pandemic began, children and teens are experiencing depression and anxiety more than ever before, according to Schalley.
</item 76>

<item 77>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
FIRST SUNDAY OF ADVENT IS A REMINDER TO ‘WATCH’ AND PREPARE FOR CHRIST'S SECOND COMING, SAYS FAITH LEADER

There are thousands of prayers on this app. And I'm instantly transported into a sacred space, no matter where I am.

No matter what, if you live in a major city or if you live in a place that's just really busy, you can pop on your headphones and connect to the app. You start
praying with a meditation and you're instantly just transported.

This is something that I've been so grateful for — and especially as we go into the hubbub and the busyness of the Christmas season, it's something that I'm
relying on daily, to be able to just cut through all that noise.

‘Transformative for people’

Question: Are people reaching out to you about your involvement with the app and asking you more about it?
</item 77>

<item 78>
Christmas proves that love triumphs over hate — even amid war, suffering, tragedy and turmoil (FOX News - Lifestyle, 2023-12-23T14:20:43+00:00)
Emanuel means "God with us." That means because of Christmas, a new, category-busting power has come into the world, one not available for the tens of thousands
of years of human existence before.

Grace and a love that surpasses all understanding is what became available because of Christmas.

The story of Longfellow's famous poem is now the subject of a feature-length film, "I Heard the Bells," available on demand through most cable and streaming
platforms.

To an age of AI and computer technology, a little story about a 19th century poet may seem anachronistic.

Longfellow's poem still bears the truth because it was based on divine reality.

But bestselling author and nationally syndicated columnist Cal Thomas notes that from his standpoint, things never really change.

Said Thomas, "You can change hairstyles, clothes, styles, modes of transportation. You can even change politicians. But nothing really seems to change … A lot
of that has to do with human nature."
</item 78>

<item 79>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
"Closeness with God is reflected and expressed in our relationships, the laws of the land, and morality. Real trust is essential in a vibrant relationship with
the living God, not merely lip service."

Survey after survey, however, along with countless anecdotal examples, paint an unmistakable picture of an America growing less religious and more secular.

The United States, it appears, no longer trusts God.

Many Americans and faith leaders believe the corresponding social decay, breakdowns of family, education, national heritage and basic science are the inevitable
outfall of a society that’s lost its guiding light.

"Closeness with God is reflected and expressed in our relationships, the laws of the land, and morality." - Pastor Bradley

Trust in God in recent decades appears to have been replaced by false idols of celebrity, technology and government.

"We have forsaken living water and created idols that can’t even hold water," said Pastor Bradley.

"America needs to come to her senses and realize the goodness of God."
</item 79>

<item 80>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
During prolonged periods of fasting, cells undergo a process called autophagy, which involves removing and recycling damaged cellular components, Antoun said.

CLICK HERE TO GET THE FOX NEWS APP

"This process is essential for maintaining proper cellular function and can help protect the brain from neurodegenerative diseases such as Alzheimer’s," he
explained.

More than six million Americans of all ages have Alzheimer's, and every 67 seconds, someone in America develops the disease, according to the Alzheimer’s
Association.
</item 80>

<item 81>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The inflation bill is responsible for about a quarter of that projected decrease.)

“He brags about how he thinks we should consider him a climate champion because he reentered the Paris accord. That Paris Agreement will take us to about 3
degrees Celsius of global heating,” Kalmus said. “I don’t think we’ll have a civilization at 3 degrees Celsius.”

(A 2022 United Nations report estimated global temperatures would rise between 2.1 and 2.9 degrees Celsius by 2100 if countries held to their climate
commitments. Many countries remain off that pace.)

Climate change is weighing on scientists, but also everyday Americans.

A 2022 poll found almost two-thirds of Americans say they have been affected by extreme weather they believe was at least partially due to climate change.

About 27% of Americans say they are “very worried” about climate change; another 27% just avoid the subject as best they can. One in 10 reported feeling
symptoms of anxiety or depression over climate change.
</item 81>

<item 82>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Whenever you’re able, suggest audio-only meetings that don’t require screens so people can do things they would naturally do in an in-person setting, like move
around or look out a window," she advised.

"Walking away from our screens can reduce the urge to multitask, reduce eye fatigue and increase our capacity to move around."

Turning off the self-view option can also be helpful, Corcoran said — "it can be particularly distracting and draining because we’re not used to seeing our own
images so much."

As the use of video platforms is still relatively new for many professionals, Corcoran noted that the long-term effects aren’t yet known.

CLICK HERE TO GET THE FOX NEWS APP

"That being said, we know that constant feelings of fatigue, exhaustion and physical strain or tension do correlate with negative mental health outcomes," she
said.

"It’s important to take steps to reduce Zoom fatigue before it leads to things like chronic stress or burnout."
</item 82>

<item 83>
'Angels, demons, spirits and souls do exist,' says exorcist priest who warns against Ouija board use (FOX News - Lifestyle, 2023-10-30T09:45:20+00:00)
"The victims were left with night terrors, suicidal ideation, despair — and one man was impaled with chicken bones in his leg that flew across the kitchen," he
said.

"Stay away from all forms of occult practices — and stay close to God."

"Once the deliverance prayers were prayed, and the participants renounced all participation with the demons, the demonic activity ceased," he also said.

"Whenever we seek to engage demons, we move away from the One True God," said Reehil.

CLICK HERE TO GET THE FOX NEWS APP

Added the priest, "Stay away from all forms of occult practices — and stay close to God."

For more Lifestyle articles, visit www.foxnews.com/lifestyle.
</item 83>

<item 84>
'In God We Trust' essential to America's future as national faith appears to wane, says Washington pastor (FOX News - Lifestyle, 2023-09-26T09:10:29+00:00)
Some reports indicate that the phrase "In God We Trust" was used by Pennsylvania troops in the American Revolution and again by the state’s soldiers in the
Civil War.

"Then conquer we must/when our cause is just/and this be our motto/in God is our trust," Francis Scott Key wrote in the now largely unknown final stanza of the
poem that became the National Anthem.

"I believe we are at a crossroads spiritually." — Pastor Bradley of Washington

The phrase first appeared on coins from the U.S. Mint in 1865 at the end of the Civil War.

"In God We Trust" was added to paper currency in 1957, after it was made the official national motto.

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

"In God We Trust" on currency has been deemed constitutional in numerous rulings by the Supreme Court and lower courts.

The U.S.
</item 84>

<item 85>
Are NBA Christmas Day Games a Better Tradition Than NFL Thanksgiving Day Games? (Essentially Sports, 2023-12-25T21:27:33+00:00)
Are the NFL’s new decisions threatening the NBA’s Christmas Day dominance?

The enduring success and importance of Christmas Day games suggest that this 77-year-old tradition will continue for the foreseeable future. However, the NFL’s
recent decisions could threaten the NBA’s stronghold on Christmas. Traditionally, the NFL has avoided Christmas Day games that fall outside of their
broadcasting window. However, last year the NFL averaged 22.1 million viewers on Christmas Day games, and 2023 could be even bigger.

via Reuters REUTERS/Kai Pfaffenbach |

The NFL has lined up a mouth-watering slate for Christmas. Superstars like Patrick Mahomes, Brock Purdy and Jalen Hurts are all set to take the field on
Christmas. According to some estimations, the NFL’s total viewership during the festive period could outperform that of the NBA by almost 75 million viewers.
This creates a huge dilemma for the NBA, as traditionally they have maintained a firm grip on Christmas.
</item 85>

<item 86>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 86>

<item 87>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
And if we can solve it, it will do so much good. It really, really will because we will find out we have more in common than not,” he said.
</item 87>

<item 88>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
“One of the most important aspects of a healthy sleep schedule is waking up around the same time each morning, so try to limit yourself to a one-hour window if
you need to adjust when you get up in the morning and when you go to sleep at night. One hour sounds like an insignificant amount of time, but it can completely
throw our circadian rhythm, or more simply known as internal clock, off balance,” he says.

Despite this, a spokesperson for Sleep School, a science-based sleep education and therapy service, told The Independent that it is “impartial” on the extent to
which the clocks changing impacts sleep, arguing that the important factor is the duration of time asleep and a consistent schedule.

Fewer depressive episodes

A Danish study from 2017 of more than 185,000 people over 15 years found that the transition from summer time to standard time was associated with an 11 per
cent rise in depressive episodes. Conversely, the switch from winter to summer time was found to have no effect.
</item 88>

<item 89>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Because fasting-mimicking diets can be done in longer periods, they can help the body go into higher levels of ketosis, Antoun noted, "which is wonderful for
the brain."

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

(Ketosis, which is when the body burns fat for energy instead of glucose, has been shown to have cognitive benefits.)

"Prolonged fasting can also help with leaky gut, which then can aid the brain," added Antoun.

In addition, intermittent fasting can help to control blood sugar , which can slow down the progression of Alzheimer's, the doctor said.

Over the past several decades, there has been a steady increase in diagnoses of Alzheimer's disease, Antoun noted.

"However, when we are able to detect neurodegenerative conditions early, we can take positive steps that may slow the process and mitigate any major side
effects of the disorder," he said.
</item 89>

<item 90>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
To nap or not to nap?

The siesta, a brief afternoon nap, is a traditional habit in Mediterranean countries. But scientists are undecided as to whether it is a healthy practice.

In the current study, specific napping habits were found to be associated with higher risk of mortality and cardiovascular disease. However, and similar to the
case of olive oil consumption, the researchers say they did not have detailed information on frequency and duration of naps or sleep structure, such as how
often and for how long, which they say weakens their conclusion.

“Moreover, reverse causation is also possible, with people napping because of subclinical conditions or other chronic diseases (for instance, if they had more
prevalence of sleep apnea), especially since napping faces a stigma in Anglo-Saxon countries,” said Sotos-Preito.

”Previous studies have found a J-shaped relationship — a smile-like curve — between napping and cardiovascular diseases,” said Sotos-Prieto.
</item 90>

<item 91>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Live your life healthier and happier with our free Living Well newsletter, packed with wellness advice, practical tips and nutritious recipes Live your life
healthier and happier with our free weekly Living Well newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to
be emailed about offers, events and updates from The Independent. Read our privacy notice Thanks for signing up to the

Living Well email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Winter is coming, as the clocks go backwards one hour at 2am on Sunday 29 October.

This will give us an extra hour in bed as we prepare for the colder months when the sun sets earlier and days get significantly shorter.

The clocks won’t go forward again until Sunday 31 March 2024.

Even so, the debate as to whether we should continue to comply with Daylight Saving Time (DST) protocols continues to run rife.
</item 91>

<item 92>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The goal of the restricted eating is to restore the circadian rhythm, the researchers said, which could help counteract the numerous disruptions that
Alzheimer’s patients face when it comes to sleep and schedule-related confusion.

HIDDEN BELLY FAT COULD SIGNAL ALZHEIMER’S DISEASE RISK 15 YEARS BEFORE SYMPTOMS SHOW UP, STUDY FINDS

"Our study emphasizes the power of feeding timing in aligning the circadian clock and its impact on the brain," said senior study author Paula Desplats, PhD,
professor in the Department of Neurosciences at UC San Diego School of Medicine, in a statement to Fox News Digital.

"We were hopeful to see some improvements in pathology, but did not expect such profound effects in reducing plaques and inflammation and improving memory," she
went on.

"The effects of time-restricted feeding on lower amyloid pathology were also measurable in blood using markers found in the clinic, which was another important
finding."

The researchers hope that these findings will lead to human trials.
</item 92>

<item 93>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

As Thanksgiving fast approaches, Americans are in the midst of writing their menus and buying the ingredients for their lavish feasts. And although it’s a day
filled with food and family, there is another popular tradition that’s held around the holiday: Friendsgiving.

Every year, Thanksgiving is celebrated on the fourth Thursday of the month, with the holiday falling on 23 November this year.
</item 93>

<item 94>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
However, the challenge lies in deciphering which elements of the Mediterranean lifestyle contribute most significantly to improved health. Sotos-Prieto and her
team designed and developed a Mediterranean lifestyle index called MEDLIFE using a questionnaire that calculated an individual’s adherence to a Mediterranean
way of life. This index considers three primary components: food consumption, dietary habits, and physical activity, including rest, social habits, and
conviviality.

This type of analysis allowed the researchers to dissect the impact of each of these components, even drilling down to specific items like “hours of sleep” or
“socializing with friends”. Moreover, the MEDLIFE index could help capture synergistic effects of multiple components that may not be apparent when examining
individual components in isolation.

In their study, the team evaluated the impact of a Mediterranean-like lifestyle over the course of ten years in more than 110,000 participants located in the
UK.
</item 94>

<item 95>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
It was, in his word, “crazy.” First, highest-version Poyer addressed his desire to understand himself better.

Advertisement

“He was basically saying, ‘Jordan, look at your life, bro — what are you mad at?’ ” Poyer says. “It was what I needed to hear because I wasn’t appreciating
anything — not my wife, my daughter, my family, my house. It was all about me and what I wanted.”

Then he turned to loving better. It was, he says, as if a Pandora’s box opened on how to love.

The remainder of the ceremonies, which lasted nine hours each, were about curiosity, energy and connectedness.

“It’s about how to be a good human,” he says. “And it grounds you. Really grounds you.”

Jordan Poyer is the Bills’ second-leading tackler but hasn’t made as many big plays as he would like for a defense plagued by injuries.
</item 95>

<item 96>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
But God hasn't forsaken us. He's not abandoned us. He's faithful to us to the very end. And it's a matter of tapping into that faith and allowing Him to be more
involved in our lives — and surrendering is the first step to getting there.

And then, to continue to cultivate that relationship through prayer — that's how we sustain that surrender.

The Hallow app and especially this Advent challenge is a beautiful way to kickstart that, especially if a person hasn't ever tried praying consistently before.

‘So many options’

Question: For those who haven't used it yet, can you describe how the app is not a replacement for Scripture but actually offers robust access to the Bible?

Roumie: I narrate most of the New Testament in the app, so you have access to the Bible here, through many works and passages of Scripture and through [other
people] as well.

And if you want something that's very specific, it is here.
</item 96>

<item 97>
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
</item 97>

<item 98>
Ultraprocessed foods with ‘feel-good chemicals’ could be as addictive as cigarettes and drugs, study suggests (FOX News - Health, 2023-10-19T09:30:48+00:00)
Additionally, Freirich recommended filling the majority of meals with minimally processed foods , as consuming more of these may result in eating fewer
ultra-processed foods.

CLICK HERE TO GET THE FOX NEWS APP

"Lastly, speak to a dietitian about how to get started on changing your diet for your personal health goals," she said.

Fox News Digital reached out to the University of Michigan researcher for additional comments.
</item 98>

<item 99>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Mackenzie Parks, a licensed clinical social worker at Children's Nebraska, said the families she encountered were "touched and flattered" that the hospital
would offer follow-up support.

"The letters I’ve received in return are really telling about the impact the program has on their well-being and hope for the future," she told Fox News
Digital.

"I also feel that it helps families know that we truly do care about them when we’re assessing them — not just because it’s our job, but because we really want
them to be safe," Parks added.

TEEN GIRLS SPEND MORE TIME ON ‘SENSITIVE’ SOCIAL MEDIA CONTENT THAT CAN HARM MENTAL HEALTH, REPORT SAYS

Eva Doescher, a licensed clinical social worker in Nebraska, shared similar feedback on her past Caring Contacts experiences.

"Caring Contacts gives me a chance to slow down and learn from the powerful, short moments that I get to connect with patients and their families," she told Fox
News Digital.
</item 99>

<item 100>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT? JOB EXPERTS REVEAL SECRETS

Corcoran agreed, noting that many report feeling livelier, happier and more active during in-person meetings.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions," she told Fox News Digital.

"As social beings, we’re highly attuned to these cues, and it takes more work for us to look for and process them in the virtual environment, leading to the
fatigue we experience."

Even beyond the workplace, Elliott noted that people are using video in more personal contexts than ever before, such as sharing reels on social media and
calling loved ones on FaceTime.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions."

"This can further exacerbate the problem if you're already spending a significant portion of your day on Zoom," he said.
</item 100>
=== END ITEM id=68967dd2fddc2e30 ===

=== ITEM id=8f6a31d9030b0a51 ===
Gold sentences and facts:
Sentence 0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 0.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 0.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 0.2: The defense argued that the FTX founder acted in good faith.
  Fact 0.3: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 0.4: The defense argued that the FTX founder basically got in over his head.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 1.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 1.5: Caroline Ellison claimed she did this under the instruction of SBF.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 3>

<item 4>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 4>

<item 5>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 7>

<item 8>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 9>

<item 10>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 10>
=== END ITEM id=8f6a31d9030b0a51 ===
