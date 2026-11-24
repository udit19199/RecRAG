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

This file has 4 items.

=== ITEM id=a81b3ac0c14fdb26 ===
Gold sentences and facts:
Sentence 0: Research shows that even a small reduction in sleep has been shown to impact your immune system, however.
  Fact 0.0: Research shows that even a small reduction in sleep impacts the immune system.
Sentence 1: “This sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this increased risk in women,” Dr Browning says.
  Fact 1.0: Dr Browning says this sex difference in insomnia emerges after puberty.
  Fact 1.1: Dr Browning says the sex difference suggests hormonal differences may be the cause of the increased insomnia risk in women.
  Fact 1.2: Dr Browning names menstruation, pregnancy and menopause as examples of hormonal differences.
  Fact 1.3: Dr Browning says hormonal differences such as menstruation, pregnancy and menopause may cause this increased risk in women.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
The minds are still frazzled. The bodies are screaming. But chaos and confusion has never felt quite so sweet.
</item 2>

<item 3>
The obsessive tormenter who made professors’ lives miserable (The Verge, 2023-10-25T13:00:00+00:00)
“A lot of the people we talk to,” she said, “have their hands in the air.”

Sinha fell asleep, as he does every night, the echoes of S.’s comments still in his subconscious.
</item 3>

<item 4>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Globally, it is estimated that around 10 per cent of adults meet the criteria for insomnia; the sleep disorder is characterised either by difficulty falling
asleep or staying asleep to such an extent that it disturbs your ability to function during the day.

Insomnia lies at the heart of the new Apple TV+ comedy series Still Up. The show follows Lisa (Antonia Thomas) and Danny (Craig Roberts), two best friends who
stay up late talking to each other because they can’t sleep. Lit by the blue glow emanating from their electronic devices, Lisa and Danny form a deep connection
over their nocturnal calls, with their friendship gradually drifting into something more. At one point, Lisa checks herself into a sleep clinic in London,
telling Danny she’s willing to “try anything” at this point – a sentiment the sleepless know all too well.
</item 4>

<item 5>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
Scientists unravel the factors of the Mediterranean lifestyle that lead to reduced mortality and how it can be adopted in non-Mediterranean populations.

Picture this: Olive oil, tomato, garlic, and an abundance of fruits — a delightful communal meal followed by a short nap. Throw in some sun and moderate warm
temperatures and you might find yourself transported to the idyllic Mediterranean.

But what if I told you that you bring the benefits of the Mediterranean lifestyle home with you, even if you live far from the region? Just like a cherished
souvenir brought back from holidays.

“The Mediterranean lifestyle represents a traditional way of living based on a plant-based diet rich in fruits, vegetables, legumes, nuts, whole grains, and
healthy oils (e.g., olive oil), moderated intake of fish, dairy and very low in red and processed meats or sweets along with food habits such as adding spices
instead of salt,” explained Mercedes Sotos-Prieto, assistant professor at the Department of Preventive Medicine, Public Health and Microbiology at the
Universidad Autónoma de Madrid.
</item 5>

<item 6>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
To nap or not to nap?

The siesta, a brief afternoon nap, is a traditional habit in Mediterranean countries. But scientists are undecided as to whether it is a healthy practice.

In the current study, specific napping habits were found to be associated with higher risk of mortality and cardiovascular disease. However, and similar to the
case of olive oil consumption, the researchers say they did not have detailed information on frequency and duration of naps or sleep structure, such as how
often and for how long, which they say weakens their conclusion.

“Moreover, reverse causation is also possible, with people napping because of subclinical conditions or other chronic diseases (for instance, if they had more
prevalence of sleep apnea), especially since napping faces a stigma in Anglo-Saxon countries,” said Sotos-Preito.

”Previous studies have found a J-shaped relationship — a smile-like curve — between napping and cardiovascular diseases,” said Sotos-Prieto.
</item 6>

<item 7>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
It’s not like how the watch uses its EKG capabilities to detect atrial fibrillation. On consumer wearables, SpO2 features generally have no medical use cases.
They’re essentially for “entertainment” purposes, and we’ve repeatedly reported that you should never, ever use them as any sort of replacement for traditional
pulse oximeters. The only real blow for Apple is if the rumors are correct and it’s working on sleep apnea detection. Losing the sensors in that instance would
make developing this feature harder. However, that feature would require a lengthy FDA clearance process. That means more advanced health features like sleep
apnea detection are more of a medium to long-term concern — and it’s very possible that Apple will have a workaround by the time such features are ready for
consumers.

That said, for this route to be successful, Iancu says it depends on the specific patent, ITC ruling, and what exactly Apple has planned as far as redesigning
the product.
</item 7>

<item 8>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Previous studies have supported this association between fasting and reduced Alzheimer’s risk.

Last year, a study led by the USC Leonard Davis School of Gerontology in Los Angeles also found that mice that followed a fasting diet showed lower levels of
"plaques and tangles" in the brain, reduced brain inflammation and better performance on cognitive tests.

"Cycles of a fasting-mimicking diet (FMD) had strong effects against Alzheimer’s symptoms in two different types of Alzheimer’s mouse models," the lead author
of that study, USC biogerontologist Dr. Valter Longo, told Fox News Digital.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain."

"In mice, the effects of FMD cycles were very strong and wide, meaning they affected both learning and memory and also Alzheimer’s pathology peptides/proteins
(amyloid and tau)."
</item 8>

<item 9>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
The Society has urged the UK government to adopt British Summer Time on a permanent basis.

“RoSPA is in favour of this proposal, and is calling for the government to adopt British Summer Time (GMT+1) all year,” the society states.

Dark, icy roads can be dangerous (Getty)

“This would mean road users will no longer experience the sudden onset of darkness during their autumn commutes, potentially saving many lives.”

And they’re not alone. The Royal Society for the Prevention of Accidents also supports the call to abolish DST, citing figures from the RAC Foundation which
state that road traffic collisions increase by 19 per cent in the fortnight after putting the clocks back one hour from British Summer Time, reducing by 11 per
cent when we put the clocks forward.

Increased immunity

Getting a good night’s sleep during the winter months is essential in protecting us against the seasonal increase in colds and flu. Research shows that even a
small reduction in sleep has been shown to impact your immune system, however.
</item 9>

<item 10>
Cell phone shocker as 97% of kids use their device during school hours and beyond, says study (FOX News - Health, 2023-10-01T09:05:26+00:00)
A whopping 97% of kids used their phones during school hours, while nearly 60% used them overnight — between the hours of midnight and 5 a.m. — on school
nights.

Social media such as Instagram and Snapchat took up most screen time, at 32% during the school day, followed by YouTube and gaming, the study found.

KEEPING KIDS SAFE: 'WAIT UNTIL 8TH' PLEDGE AIMS TO EMPOWER PARENTS TO RESIST SMARTPHONES FOR CHILDREN

Overnight phone use was primarily spent engaging with the same media, although YouTube appeared to be the longest-running app because videos were often left
playing during the night.

Teens spent an average of 20 minutes per night on their phones, while some spent up to five hours.

Sixty-seven percent of participants picked up their phones on school nights, but the median number of pickups per night was only once.
</item 10>

<item 11>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 11>

<item 12>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 12>

<item 13>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 13>

<item 14>
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
</item 14>

<item 15>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The symptoms of insomnia can last a long time; a longitudinal study conducted in the UK found that 69 per cent of respondents affected by the sleep disorder at
the start of the study still had insomnia 12 months later.

This may go to show how complex treating insomnia can be. Explaining the appeal of quick fixes, Dr Browning says, “In the short term you’re changing something,
so [they] can often [work] like a placebo. If hunger is keeping you awake, eating before bed will help. If your mattress is uncomfortable, sleeping on the floor
will help.” But the problem isn’t just that these methods don’t necessarily work – it’s that they could be used to exploit vulnerable insomniacs.

“I’m a sleep therapist, my doctorate is in insomnia and it’s very, very frustrating,” she says. “People who have insomnia are desperate.
</item 15>

<item 16>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
This type of chart illustrates situations in which low values of one variable (in this case, a short nap duration) lead to a decrease in the values of the
second variable (incidence of cardiovascular events), while the trend reverses at higher values.

“Short naps, which range from 0 to 30 minutes, were associated with lower incident cardiovascular disease,” said Sotos-Preito. “An inverse association between
short naps and mortality and cardiovascular diseases was found in our previous publications.”

When asked for a definitive answer as to whether napping was indeed beneficial or not, Sotos-Prieto indicated that more research is needed in this regard.

Adapting to the lifestyle

The challenge remains in how to put these results into practice in other countries with different, deep-rooted customs. Cultural adaptation of beneficial
practices is essential to make the Mediterranean diet and habits more appealing and practical for diverse populations in non-Mediterranean countries.
</item 16>

<item 17>
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
</item 17>

<item 18>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Lavender oil. Eating kiwis. Masturbation. Everyone has their own “hacks” when it comes to falling asleep in 2023; the days of counting sheep are long gone.
Jennifer Aniston’s top tip for battling insomnia is to “sleep with my phone at least five feet away from me”, while Martha Stewart recommends focusing on your
breathing.
</item 18>

<item 19>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications (Getty Images)

As with many other mental health disorders, therapy can often provide long-term relief from insomnia. The most effective of these therapies is Cognitive
Behavioural Therapy for Insomnia (CBTI),” says Dr Browning, describing it as a “complete cure” that a lot of people don’t know about. “Different from CBT for
depression or CBT for anxiety, CBTI works by addressing the negative thoughts and worries that perpetuate insomnia.”

Other avenues of treatment are also proving increasingly fruitful. A new drug called Quviviq was just approved by the NHS last week. Dr Browning is optimistic
about the “exciting” new medication, explaining that it acts in a very different way to other sleeping pills, which “knock you out” but cause people to develop
a tolerance.
</item 19>

<item 20>
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
</item 20>

<item 21>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“As much as CBTI is the gold standard across the world, there might be some people who need a bit of [medicinal] help as well, a bit like depression.”

In the meantime, Dr Browning says there’s no harm in trying the occasional Reddit recommendation – if it’s free. That might be an act of self-pleasure,
narrating the plot of Pride and Prejudice in your head, or listening to shipping forecasts. FaceTiming your best friend into the early hours of the morning –
like Lisa and Danny do in Still Up – probably isn’t encouraged, but it does make for a very good romcom.

‘Still Up’ is available for streaming on Apple TV+.
</item 21>

<item 22>
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
</item 22>

<item 23>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 23>

<item 24>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 24>

<item 25>
We could end the AIDS epidemic in less than a decade. Here's how. (Live Science: The Most Interesting Articles, 2023-12-01T12:00:17+00:00)
Eastern Europe and Central Asia also lag far behind, with just 51% of people with HIV getting ART and less than half being virally suppressed.

Related: Oldest 'nearly complete' HIV genome found in forgotten tissue sample from 1966

Beyond 2030

We face many obstacles on the road to ending the AIDS epidemic — but we do hold all of the tools to get there, Abdool Karim, Sharma, Gandhi and Landovitz
agreed. By using those tools effectively, we could begin to meaningfully drive the number of new HIV infections toward zero. At that point, HIV would become a
manageable, chronic disease of the elderly.

Already, about a quarter of people with HIV worldwide, and about half of adults with HIV in Western and Central Europe and North America, are at least 50 years
old. "They're growing older with HIV; they're not dying from HIV or AIDS," said Sharma, whose research focuses on aging populations with HIV.
</item 25>

<item 26>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Long-time insomniac and video producer Preshita says she has tried many tricks over the years – “from drinking chamomile tea to counting backwards from 1,000”.
After much experimenting, she has found that taking a shower before bed and keeping her bedroom clutter-free were the most effective methods. “Someone
recommended masturbating, and that was another thing that worked,” she adds.

‘Still Up’ follows Lisa (Antonia Thomas) and Danny (Craig Roberts) as best friends who stay up late talking to each other because they can’t sleep (Courtesy of
Apple)

Millions of people struggle with insomnia in the UK alone. Studies have found that women are 40 per cent more likely to be affected by insomnia than men. “This
sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this
increased risk in women,” Dr Browning says.
</item 26>

<item 27>
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
</item 27>

<item 28>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
Attending meetings in pajama bottoms might seem like a job perk, but some remote workers have found that videoconferencing for work isn’t all it’s cracked up to
be.

In the years since the COVID pandemic triggered a spike in telecommuting, many have complained of so-called "Zoom fatigue."

Thought to be coined by Stanford University professor Jeremy Bailenson, the term refers to a level of exhaustion that comes with interacting with others on
camera all throughout the day.

REMOTE WORKERS RESIST MONDAY OFFICE HOURS, BUT BUILDING 'ORGANIZATIONAL CULTURE' IS KEY, SAYS EXECUTIVE

Dr. Kyle Elliott, a tech career coach in San Francisco, defines Zoom fatigue as "incessant tiredness or lack of energy as a result of spending a large quantity
of time on Zoom."

Additional symptoms can include physical tension (including headaches, eye strain and stiffness), moodiness, a sense of disengagement or difficulty
concentrating, according to Dr.
</item 28>

<item 29>
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
</item 29>

<item 30>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 30>

<item 31>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Many people are concerned about spiking energy bills this winter (Getty/iStock)

During winter, evening energy demand peaks between 5pm and 7pm, with UK households potentially facing three-hour power cuts this winter, the National Grid has
warned, due to a shortage of gas.

“We are no longer in an energy crisis in Europe but an energy war and, dependent on weather conditions this winter, it is very likely we may need to start
rationing energy very seriously to avoid bigger energy issues in December and January when gas reserves start to run low,” adds Professor Foley.

“There is no doubt that by foregoing the DST in winter we would save a lot of energy, reduce our bills and carbon emissions during this energy war, especially
during a cost of living crisis.”

Fewer road accidents

The Royal Society for the Prevention of Accidents (RoSPA) has called for the abolition of DST, arguing that turning the clocks back increases the risk of road
accidents, due to more people driving during darker winter months.
</item 31>

<item 32>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Being social, says Laird, is as important as trying new things: “Research shows that those who are more social are also more resilient to chronic conditions and
often have a stronger immune system.”

Anyone with older parents should be urging them out for a walk five days a week Dr Eamon Laird, health and ageing researcher

Depression among the elderly is on the rise and often goes unnoticed. Gerontologists now believe that it is one of the most common psychiatric disorders among
the elderly – with half of all cases going undiagnosed. When elderly parents become depressed (many of them come from a generation not accustomed to talking
about their mental health), it can be difficult to know how to respond. But Laird’s latest report suggests that a daily walk could help. His study of more than
4,000 older adults found that as little as 20 minutes of daily walking could cut the risk of depression by up to 43%.
</item 32>

<item 33>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And there are indications that Apple is actively pursuing that option. Bloomberg reports that Apple engineers are scrambling to alter the blood oxygen
algorithms on the watches. The report quotes an unnamed Apple spokesperson saying the company plans to submit a workaround to the US customs agency to get the
product back on shelves.

Do Apple’s SpO2 sensors even matter? Speaking as a wearables reviewer, removing the SpO2 sensors wouldn’t be the end of the world for Apple in the short term.
Apple’s blood oxygen features are mostly for wellness and sleep tracking — and they’re not that useful beyond determining your nightly baseline. I’m willing to
bet the average Apple Watch user forgets that the blood oxygen features even exist. In the case of the Apple Watch, it periodically takes SpO2 readings in the
background, or you can actively take a spot check. However, Apple itself doesn’t really make wide use of these metrics. It’s not like how the watch uses its EKG
capabilities to detect atrial fibrillation.
</item 33>

<item 34>
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
</item 34>

<item 35>
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
</item 35>

<item 36>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
“There’s also a big drop in suicides and car crashes worldwide. That all adds up to tens of thousands of fewer deaths around the world from just one extra hour
of sleep.”

When did Daylight Saving Time begin?

The UK first introduced Daylight Saving Time in 1916 when the government introduced British Summer Time (BST) to encourage people to spend more time outdoors in
daylight.

The change hasn’t always proved popular, however, with the European parliament voting in favour of scrapping DST in 2019, a change that was due to take effect
for the first time in 2021, but plans have been stalled.

A 2019 YouGov poll found that the majority of Brits were marginally in favour of keeping DST, however, with 44 per cent electing to keep the current system,
while 39 per cent voted in favour of ditching the change.
</item 36>

<item 37>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Live your life healthier and happier with our free Living Well newsletter, packed with wellness advice, practical tips and nutritious recipes Live your life
healthier and happier with our free weekly Living Well newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to
be emailed about offers, events and updates from The Independent. Read our privacy notice Thanks for signing up to the

Living Well email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Winter is coming, as the clocks go backwards one hour at 2am on Sunday 29 October.

This will give us an extra hour in bed as we prepare for the colder months when the sun sets earlier and days get significantly shorter.

The clocks won’t go forward again until Sunday 31 March 2024.

Even so, the debate as to whether we should continue to comply with Daylight Saving Time (DST) protocols continues to run rife.
</item 37>

<item 38>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 38>

<item 39>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Right now, the current plot is about the centralization of discourse on a few corporate platforms. The rise of the platforms brings potential to network
democratic uprisings, as well as buoy authoritarian leaders through post-truth memes and algorithms optimized to dish out anger and hatred. This is a more
challenging story to narrate, because the setting is everywhere. The story unfolds in our bedrooms while we should be sleeping or waking up, filling the most
quotidian moments of waiting in line in the grocery store or while in transit. The characters are us, even more intimately than with climate change. It makes it
hard to see the shape and meaning of this story. And while we are increasingly aware of the influence that shifting our media and social lives onto big tech
platforms has on our democracy, less attention is devoted to the influence this has on our ability to respond to climate change.

Think about these three forces meeting — climate change, authoritarianism, the Internet. What comes to mind?
</item 39>

<item 40>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Many academics, scientists and campaigners have called for the abolition of the clock turning-tradition, citing improvements to mental and physical health,
public safety and wellbeing.

But what are the advantages of such a decision? We speak to the experts to get the low-down on the issue.

Annual energy savings of £400

Professor Aoife Foley from Queen’s University Belfast has calculated that household consumers could save more than £400 per year from their electricity bill
depending on their tariff if the government scrapped DST at the end of October.

Dr Foley, who specialises in clean energy research, estimates that we could save £1.20 a day and dramatically reduce demand on the National Grid.

“By simply foregoing the winter DST in October, we save energy because it is brighter in the evening during winter, so we reduce commercial and residential
electrical demand as people leave work earlier, and go home earlier, meaning less lighting and heating is needed,” she said.
</item 40>

<item 41>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“At this point, you’re no longer worried about the traumatic incident; you’re now worried about not being able to sleep.” It’s a self-fulfilling prophecy.

Social media consultant and mother of two, Amy Fieldhouse-Downes, started suffering from insomnia after the birth of her second son in 2019. She believes the
affliction was rooted in anxiety. “If you Google how to go to sleep, I’ve probably tried everything on there,” she says. “Going to bed earlier, going to bed
later, meditations that would get cut off by an advert, lavender products, sleep cream, eye masks, Sominex sleep tablets. And I even used it as an excuse to
have a few glasses of wine.” It wasn’t until she tried sleeping with specialised earplugs that she was finally able to get some rest.
</item 41>

<item 42>
10 scorching-hot discoveries made about the sun in 2023 (Live Science: The Most Interesting Articles, 2023-12-24T14:00:00+00:00)
Back when the current solar cycle began in 2019, forecasts from space weather scientists suggested that the solar maximum would arrive sometime from 2025 onward
and be relatively weak compared to past solar cycles. However, numerous warning signs early in the year, including rising sunspot numbers , powerful X-class
solar flares and changes in Earth's upper atmosphere , hinted that the solar cycle was not progressing as expected.

In June, Live Science published an exclusive, in-depth feature on the upcoming solar maximum, which revealed that the explosive peak would likely arrive by 2024
and be more active than the initial forecast suggested. And in October, space weather scientists released an updated forecast for the current solar cycle, which
is more in line with what experts told Live Science.

When solar maximum does arrive, we can expect to be bombarded with more solar storms, which will result in frequent auroral displays , radio blackouts,
potential satellite problems and disruptions to the migration patterns of some animals .
</item 42>

<item 43>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
The shorter format is an opportunity to focus and update the information — there have been quite a few discoveries in lots of different aspects of deep sea
biology since I wrote "Ask an Ocean Explorer."

We know far more about the deep ocean and its inhabitants than about the moon or Mars, says Jon Copley. (Image credit: Giordano Cipriani via Getty Images)

SP: Research has made great strides in recent years, I'm sure. What are some of the most exciting, new discoveries you discuss in the book?

JC: We're finding out a lot more about how deep sea animals interact with each other and their environment. An area where we've seen a lot of interesting papers
over the past five years has been in sensory ecology — realizing how animals perceive their environment, how they respond to that, how they avoid being seen by
predators... It was nice to bring some of those together in a couple of the chapters.
</item 43>

<item 44>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Research shows that even a small reduction in sleep has been shown to impact your immune system, however.

Making the move from daylight saving time interferes with your natural circadian rhythm, the 24-hour biological cycle that is influenced by morning light and
evening darkness, however.

“It’s hard to understate the importance of a good night’s sleep,” says Giulia Guerrini, lead pharmacist of digital pharmacy Medino. “Having too much or too
little sleep can have adverse side effects, so you should aim to get between seven and eight hours of sleep as consistently as possible.”

Never underestimate the value of a good night’s sleep (Getty)

“One of the biggest disruptors to our sleep is when UK clocks go forward one hour at 1am on the last Sunday in March and go back one hour at 2am on the last
Sunday in October, which is rapidly approaching on 30 October.
</item 44>

<item 45>
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
</item 45>

<item 46>
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
</item 46>

<item 47>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
“One of the most important aspects of a healthy sleep schedule is waking up around the same time each morning, so try to limit yourself to a one-hour window if
you need to adjust when you get up in the morning and when you go to sleep at night. One hour sounds like an insignificant amount of time, but it can completely
throw our circadian rhythm, or more simply known as internal clock, off balance,” he says.

Despite this, a spokesperson for Sleep School, a science-based sleep education and therapy service, told The Independent that it is “impartial” on the extent to
which the clocks changing impacts sleep, arguing that the important factor is the duration of time asleep and a consistent schedule.

Fewer depressive episodes

A Danish study from 2017 of more than 185,000 people over 15 years found that the transition from summer time to standard time was associated with an 11 per
cent rise in depressive episodes. Conversely, the switch from winter to summer time was found to have no effect.
</item 47>

<item 48>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The series was inspired by creator Natalie Walter’s own experience with insomnia, and her late-night Facebook chats with Steve Burge, her co-writer on Still Up.
“As part of the research for [the show] we visited a sleep clinic, which was fascinating,” she tells me. “Some evolutionary biologists think the reason humans
have different sleep patterns is that during our early history, a few members of the tribe would always be awake and vigilant during the night, and so could
protect everyone else from predators. Not that I’d be much use fighting off a woolly mammoth.”

Just like depression or anxiety, insomnia can often be triggered by something. Whether it’s a life event as monumental as becoming a new parent, or something as
commonplace as jet lag, there are any number of things that can prompt a bout of sleeplessness. The issue, explains sleep therapist Dr Lindsay Browning, is that
insomnia can often remain long after the trigger has gone.
</item 48>

<item 49>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
In her 1973 feminist novel Fear of Flying, author Erica Jong notes that, sometimes, the only way to overcome sleeplessness is by pretending you don’t care about
nodding off. “Then sleep became piqued, like a rejected lover, and crept up to try to seduce you.”

In a world where human sleep patterns have increasingly been plunged into chaos, people are having to turn to ever more creative – and often bizarre – means of
getting their 40 winks in. Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are
foolproof. (Kiwis, by the way, contain serotonin and antioxidants, which is why it’s thought they might have sleep-promoting benefits.) The result is a society
with a growing problem on its hands: persistent sleeplessness – otherwise known as insomnia.
</item 49>

<item 50>
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
</item 50>
=== END ITEM id=a81b3ac0c14fdb26 ===

=== ITEM id=51b7a599ee3728d1 ===
Gold sentences and facts:
Sentence 0: “Users’ natural behavior for search is to follow the path of least resistance, i.e.
  Fact 0.0: Users' natural behavior for search is to follow the path of least resistance.
Sentence 1: “We make Google be the default search engine,” he said, “because we’ve always thought it was the best.
  Fact 1.0: A speaker said 'We make Google be the default search engine.'
  Fact 1.1: The speaker said the reason was 'because we've always thought it was the best.'

Retrieved items:
<item 1>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
However, the startup wants subscriptions to be its primary revenue driver. Deft is currently testing two paid tiers: a $10 a month plan, which will offer better
search customization to filter out some brands, and a $20 per month plan, which will enable personal shopping service, that will continue to look for suitable
products even after you stop searching manually.

Up next, Deft aims to release a browser extension that helps users search for products by the end of this year or early next year. In the next three years, the
company wants to build and release search APIs that other e-commerce sites could use.

Challenges of building a search startup

Given that it’s building a search solution for e-commerce, Deft already faces challenges from Amazon and Google. In September, Amazon updated its visual search
feature that lets users snap a photo of a product and search for similar items on the platform. Google released its multi-search feature last year that lets
users fuse images and text.
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 5>

<item 6>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
Google released its multi-search feature last year that lets users fuse images and text. Separately, Deft will also face competition from other startups like
Objective, a Matrix Partners-backed startup building an API platform for multimodal search.

Caroline Casson, a partner at Vitalize Venture Capital and an investor in Deft, believes the startup has an edge because of its knowledge graph for products.

“Building a robust e-commerce search engine is incredibly complex on the backend. The extensive knowledge graph that Deft has built combined with AI search
capabilities makes for a high-quality user experience. We were excited about early iterations of the product when we invested and we continue to be impressed
with each product enhancement,” Casson told TechCrunch over email.

Adii Pienaar, another investor, and co-founder of WooCommerce, which was acquired by Automattic, told TechCrunch over a call that he sees Deft as solving user
experience problems.

“Over the last couple of years, consumers have been trying out new ways to interact with and purchase products.
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 10>

<item 11>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
He said that the main problem for startups in this space is the go-to-market strategy and fundraising.

“The problem with competing with e-commerce players is not just about user experience, it’s also about logistics infrastructure and supply chain. So a startup
catering to consumer search has to invest a lot of money into making it a viable business,” Konigsberg told TechCrunch.

He also added that it will be difficult for a startup like Deft to build user loyalty while competing against Google and Amazon.

Both Pienaar and Konigsberg said that Deft will have to involve retailers in the mix either as partners or as customers of their search product. Either way,
it’s going to be interesting to see if Deft can build a user base on its own or sell its search feature to e-commerce retailers.
</item 11>

<item 12>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 12>

<item 13>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
The startup said it is on track to hit 50,000 monthly active users by the end of this year. While co-founders didn’t reveal the exact search volume the product
is catering to, they said Deft is seeing “tens of thousands” of searches every week. They added that people using Deft for the second time do more searches than
their first visit.

The team also noted that it has a conversion rate that is five times higher — a measure of people buying the product versus seeing it — compared to the standard
e-commerce rate, which is around 2-3%, according to various estimates.

Deft doesn’t show ads on its platform. Rather, it relies on affiliate commissions to generate revenue. However, the startup wants subscriptions to be its
primary revenue driver.
</item 13>

<item 14>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 14>

<item 15>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 15>

<item 16>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 16>

<item 17>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 17>

<item 18>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Yahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome, which ended 2014 with nearly
50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage on desktop: and mobile is even
more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing the blame on Yahoo certainly
works in Google’s favor here!

The trial continues…
</item 18>

<item 19>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 19>

<item 20>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 20>

<item 21>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
Users can go to Deft and search for things like “wooden lamps with three legs under $400” or upload an image, and modify their search to “lamps under $400 that
look like this.” While it’s hard to know you can do this if someone doesn’t tell you that it works, users can also paste the image directly into the search bar.
The company is working on highlighting the image search feature better.

The company said that it doesn’t just look at the listing and description for search results. It also looks at materials, reviews, photos and more. This way,
when someone searches for a “pet-friendly” couch, Deft can return results with stain and scratch-resistant material.

“You have companies like Google and Amazon that continue to push the most relevant listings down so they can make more advertising dollars.” Zach Hudson

Deft is currently focusing on furniture and home décor search but aims to expand to different categories along the way. The company has built its own scraping
tech to add products to its search results.
</item 21>

<item 22>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 22>

<item 23>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 23>

<item 24>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
I see Deft as creating a new kind of UX for purchasing,” he said.

Startups like Neeva — which was founded by former Google execs — tried to convince people to switch to paid and private search with additional features.
However, earlier this year, the startup shut down its consumer efforts and eventually got acquired by Snowflake.

Pienaar said that Deft has to give compelling reasons to users to switch to its service.

“Users’ natural behavior for search is to follow the path of least resistance, i.e. most people just pick up their phone and just use the default search engine.
For Deft to change that behavior, they will need to create a high-value proposition,” he said.

Amir Konigsberg, an investor and founder of e-commerce search startup Twiggle (which shut down), said his startup wanted to solve ChatGPT-type interactions in
the shopping world. He said that the main problem for startups in this space is the go-to-market strategy and fundraising.
</item 24>

<item 25>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 25>

<item 26>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
The company has built its own scraping tech to add products to its search results. It shows results from Target, Pottery Barn and other places on the web. The
company hasn’t added searches from Amazon yet because it feels there are a lot of “junk” listings on the site and it will take some work to clean up that data.

Hudson told TechCrunch that there are a few plaguing problems with the traditional search experience that prevent users from finding a suitable item.

“You have companies like Google and Amazon that continue to push the most relevant listings down so they can make more advertising dollars. Some easy searches
are now hard because marketers have learned how to game these search engines. Plus, now you have AI content that is putting the burden on users to separate fact
from fiction,” he said.

The company raised $1.8 million in a seed round from Hustle Fund, Frontier VC, Vitalize, Long Ecommerce Ventures, 43 Ventures and angel investors.
</item 26>

<item 27>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
E-commerce search startup Deft is releasing a multimodal search function for users to get better results. The company, which opened its service to beta users in
May, has rolled out the new search experience so users don’t have to describe the item in words.

The company is trying to solve the problem of finding the right item without spending hours looking for a product. Often users end up with subpar results while
searching for products because their search query description won’t match the details of listed items. The startup is relying on different techniques, including
building its own parser and knowledge graph combined with large language models (LLMs) and multimodal search.

Deft was founded in 2019 by Alex Gunnarson and Zach Hudson. Gunnarson started working on the idea after he couldn’t find a specific type of couch online even
after putting in different parameters and descriptions. It took him hours to find a perfect result for his search, so Gunnarson decided to build a solution with
Hudson.
</item 27>

<item 28>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 28>

<item 29>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Google is now making it the default sign-in method for all users.

2 sec, let me text you some cash: When questioned about Mastercard’s prospects in emerging markets such as India, Mastercard’s CFO Sachin Mehra praised UPI for
helping with digitization but voiced reservations about its commercial sustainability, saying it is an “incredibly painful experience” for ecosystem
participants.

Bravely browsing — or searching — for a new job . . . : Brave Software, the maker of Brave Browser and Search, confirmed that it has laid off 9% of its
workforce across departments.
</item 29>

<item 30>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 30>
=== END ITEM id=51b7a599ee3728d1 ===

=== ITEM id=c614cbb84279e785 ===
Gold sentences and facts:
Sentence 0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 0.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 0.1: The defense countered that the FTX founder acted in good faith.
  Fact 0.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 0.3: The defense countered that the FTX founder basically got in over his head.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: A speaker referred to as 'she' said it was Bankman-Fried's decision.
  Fact 1.1: The speaker said Bankman-Fried was the CEO of FTX.
  Fact 1.2: The speaker said it was Bankman-Fried's decision as he was the CEO of FTX.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 4>

<item 5>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 6>

<item 7>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 7>

<item 8>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 9>

<item 10>
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
</item 10>

<item 11>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 11>

<item 12>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 12>

<item 13>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 13>

<item 14>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 14>

<item 15>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 15>

<item 16>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Joe Bankman, Bankman-Fried’s father. He is among the people included in the “small group chat,” the Signal group where the FTX inner circle strategized during
the exchange’s last days.

Barbara Fried’s political action committee, Mind the Gap. It is among the entities that received FTX customer money as political donations. Fried and Bankman
lived together in a Bahamas house purchased for them using FTX investor funds, according to a financial expert called by the prosecution; in a lawsuit filed by
FTX’s bankruptcy estate, Bankman and Fried are alleged to have been given $10 million of FTX customer funds.

Gabe Bankman-Fried’s nonprofit, Guarding Against Pandemics, which got millions from FTX. He was also in a Signal group chat, Donations Processing, that Singh
says was used to coordinate the transfer of FTX and Alameda money to various political causes. In that chat, he can be seen recommending some donations be made
“in the dark” to avoid public relations headaches.
</item 16>

<item 17>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 17>

<item 18>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
Meanwhile, Bankman-Fried tweeted. “FTX is fine. Assets are fine” and “We don’t invest customer assets even in treasuries.”

Pointing at Bankman-Fried, Rehn said, “This man stole billions of dollars from thousands of people.”

So how was the defense going to follow it up? I was very curious, having learned yesterday that Bankman-Fried had never been offered a plea deal since he and
his attorneys had told the government they wouldn’t negotiate. Surely there would be some manner of evidence, some something, that would have made him so
confident.

There was, instead, a metaphor.

Defense attorney Mark Cohen, with the energy of a patient father telling his obnoxious children a bedtime story, assured us that working at a startup was like
building a plane while flying it, and that FTX the plane had flown right into the perfect storm: the crypto crash.
</item 18>

<item 19>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating

During Yedidia’s testimony, we were shown a photograph in court: former President Bill Clinton and former Prime Minister Tony Blair onstage in suits next to
Bankman-Fried, in a rumpled T-shirt and cargo shorts. Sure, dressing down is a staple of Silicon Valley startup culture — and a sign of its swagger. The defense
counsel tried to elicit testimony from Yedidia that the cargo shorts were evidence of Bankman-Fried’s humility, but the photograph seemed to signify just the
opposite: that the man in cargo shorts is so important, he doesn’t need to dress up, even to meet former world leaders.

Now, in court, Bankman-Fried is wearing a suit.

In 2019, just months after FTX was founded, Wang oversaw code called “allow_negative” that let Alameda Research avoid being liquidated by FTX’s risk management.
It was turned on in July.
</item 19>

<item 20>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 20>
=== END ITEM id=c614cbb84279e785 ===

=== ITEM id=f3729206d6abc0d4 ===
Gold sentences and facts:
Sentence 0: Rangers also make a lot of choices during character creation (and leveling up) that determine how they play.
  Fact 0.0: Rangers make a lot of choices during character creation.
  Fact 0.1: Rangers make a lot of choices during leveling up.
  Fact 0.2: The choices Rangers make during character creation and leveling up determine how they play.
Sentence 1: During character creation, you must pick the otherworldy patron that you’ve decided to serve.
  Fact 1.0: During character creation, you must pick an otherworldly patron.
  Fact 1.1: The patron you must pick during character creation is the one you have decided to serve.

Retrieved items:
<item 1>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, Warlock, and Wizard classes.
</item 1>

<item 2>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Picking the best class in Baldurs Gate 3 is one of the most important decisions you’ll make in the game — and you have to do it right at the start, with minimal
context to work off.

Your class mostly determines how your character will fight in Baldur’s Gate, but with 12 classes — and 46 subclasses total across them all — it’s not really an
easy decision. Here’s what to know about finding the best class (and the best subclass) for you in Baldur’s Gate 3.

All BG3 classes are ‘good’

The classes in Baldur’s Gate 3 — and in pen-and-paper D&D, for that matter — are all fairly well-balanced. There’s no single best class (barrelmancer aside) in
Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the best class really just boils
down to how you want to play.
</item 2>

<item 3>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Wizards don’t get any armor and are limited to just daggers and quarterstaffs (quarterstaves?) in combat, so they’re going to rely on magic to do any damage.

Your Wizard subclass is chosen when you create the character. You’ll choose from eight schools of magic: Abjuration, Conjuration, Divination, Enchantment,
Evocation, Illusion, Necromancy, and Transmutation. Each subclass gets an extra ability themed to the school of magic. Broadly speaking, Abjuration is about
protection and defense, Conjuration is about creating something from nothing, Divination gets to foresee (and change) rolls, Enchantment is about charming and
controlling minds, Evocation shapes and controls spells, Illusion is about trickery, Necromancy controls life and death, and Transmutation focuses on alchemy
and changing the world around you.

Evocation is the best subclass for a Wizard. Consult our Wizard build guide for more details on why, and to see which skills and feats to pick.
</item 3>

<item 4>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Spell Sniper. You learn a cantrip, and the number you need to roll a Critical Hit while attacking is reduced by 1. This effect can stack.

For more information on the classes in Baldur’s Gate 3, consult our overviews on the Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue,
Sorcerer, and Wizard classes. If you’re stumped by the choice, consult our guide on what class to start with.
</item 4>

<item 5>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Is Warlock the best class for you in BG3?

Warlocks sacrifice the number of spells that they know for sheer power. Every Warlock spell slot is the same — the highest level available — and they recharge
on a short rest instead of a long rest. Warlocks aren’t as versatile as other caster classes, but what they do, they do well.

Warlocks are a great class for you if you want to play a dedicated caster class, but don’t want to deal with all the prep work that classes like Wizards
require. You’ll still get to use (a lot of) magic, but you won't have to manage a binder full of spells every morning.
</item 5>

<item 6>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Rangers get Light and Medium Armors and some of the more basic weapons.

Rangers also make a lot of choices during character creation (and leveling up) that determine how they play. At level 1, you’ll choose a Favored Enemy — a
preferred quarry — and a preferred environment that grants you some magical spells.

At level 3, you get to choose a Ranger subclass from Beast Master, Gloom Stalker, or Hunter. Beast Masters get a Companion animal that fights along side you,
Gloom Stalkers focus on stealth, and Hunters get to specialize their melee abilities.

Hunter is the best subclass for a Ranger. Consult our Ranger build guide for more details on why, and to see which skills and feats to pick.

Best class for damage: Barbarian

Barbarians are all about combat. They get light and medium armor (and shields), but they’ve also got the option to keep their AC up even without wearing armor.
</item 6>

<item 7>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They only get Light Armor, so they have to focus on staying unseen and quick. Their weapons tend to be simple and focus on ones that have the Finesse type —
meaning you can add your Dexterity modifier instead of your Strength. All Rogues get to use Dash (double movement), Disengage (don’t provoke Opportunity
Attacks), and Hide as Bonus Actions. They also get Sneak Attack combat actions that let them deal extra damage whenever they have Advantage.

Level 3 Rogues choose from the Thief, Arcane Trickster, and Assassin subclasses. Thief Rogues get an extra bonus action, Arcane Tricksters mix in magic, and
Assassins get advantages over surprised or unprepared enemies.

Thief is the best subclass for a Rogue. Consult our Rogue build guide for more details on why, and to see which skills and feats to pick.

Best class for melee and magic: Paladin

Paladins start out mixing in (Divine) magic with their melee abilities. They’re the most magic-focused of the melee classes.
</item 7>

<item 8>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
For weapons, Barbarians get a wide range of heavy hitters like battleaxes and warhammers. Every Barbarian gets the Rage ability that lets them hit even harder
and reduce damage during combat (and only during combat).

When you hit level 3, you get to choose from the Wildheart, Berserker, or Wild Magic subclasses. Wildheart lets you modify your Rage ability with animal-themed
tweaks like healing, jumping, or a stampede. Berserker is a lot more straightforward and focuses on hitting (extra) things and throwing. And Wild Magic adds a
magical flair to your Rage with a random magical effect and improved saving throws against magic for you and your allies.

Berserker is the best subclass for a Barbarian. Consult our Barbarian build guide for more details on why, and to see which skills and feats to pick.

Best class for magic: Druid

Druids are the nature-based casters of Baldur’s Gate 3.
</item 8>

<item 9>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Where Bards come into their own, though, is with Bardic Inspiration — a way to boost the rolls of their allies — and their spells that are a nice mix of utility
and offensive magic.

Bards get three subclasses at level 3: College of Lore, College of Valor, and College of Swords. College of Lore Bards get a lot of useful skills to use outside
of combat and get to use their Bardic Inspiration to make enemies weaker, College of Valor Bards help their allies hit even harder, and College of Swords Bards
focus on a swashbuckling style of combat.

College of Valor is the best subclass for a Bard. Consult our Bard build guide for more details on why, and to see which skills and feats to pick.

Best class for magic damage: Wizard

Wizards are the archetypical caster class. They are the only Intelligence-based class in Baldur’s Gate 3 and have the most spells — and the most spell slots to
cast them — of all of the casters.
</item 9>

<item 10>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
They also get a way to heal themselves and way to get a second attack once per short rest.

At level 3, you get to choose a subclass from Battle Master, Eldritch Knight, and Champion. Battlemasters get a set of Superiority Dice that let you perform
Maneuvers to control the flow of battle and the battlefield. Eldritch Knights add magic into the Fighter’s arsenal. Champions keep it simple with a better
change to deal a Critical Hit (for double damage).

Champion is the best subclass for a Fighter. Consult our Fighter build guide for more details on why, and to see which skills and feats to pick.

Best ranged class for beginners: Ranger

Rangers are agile hunters that focus on Dexterity as their primary Ability. Rangers are an interesting mix of melee, stealth, skills, and magic. That also makes
it harder to get the most out of the class. Rangers get Light and Medium Armors and some of the more basic weapons.
</item 10>

<item 11>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Best magic class for beginners: Warlock

Warlocks are casters that made a deal with a higher power for their magic. They play a lot like Sorcerers, but with a slight emphasis on more academic Skills
and fewer spells. Warlocks get Light Armor, so they’re a little less squishy than Sorcerers.

Like Sorcerers, you’ll pick your Warlock subclass during character creation. You’ll choose from The Fiend, The Great Old One, and The Archfey subclasses that
are all themed around where your magic comes from — and determine some of the spells you can use. The Fiend Warlocks get their magic from a demon-like fiend and
have hell-themed spells, the Great Old One is more Lovecraftian horror-themed, and the Archfey draws power from a powerful Fey being.

Great Old One is the best subclass for a Sorcerer. Consult our Sorcerer build guide for more details on why, and to see which skills and feats to pick.
</item 11>

<item 12>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
The Warlock is one of twelve classes in Baldur’s Gate 3. Warlocks are extreme knowledge-seekers who are willing to form a pact with otherwordly beings in order
to gain ancient secrets that’ll bolster their magical arsenal.

To deal with such powerful creatures, Warlocks need to have strong mental power. They need to be able to make a deal with the devil and survive. As such, their
primary ability score is Charisma.

There’s no single best class in Baldur’s Gate 3, and the game gives you as many opportunities for creative problem solving as it does for punching. Finding the
best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to experiment with
multiple classes at once.

Read on to find out everything you need to know about the Warlock class in Baldur’s Gate 3, and learn all about the best Warlock subclass, feats, and build.
</item 12>

<item 13>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks.

. Your Charisma Increases by 1, to a maximum of 20. Your Proficiency Bonus is also doubled for Deception and Performance Checks. Elemental Adept . Your spells
ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1.

. Your spells ignore Resistance to a damage type of your choice. When you cast spells of that type, you cannot roll a 1. Moderately Armored . You gain Armor
Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20.

. You gain Armor Proficiency with Medium Armor and shields, and your Strength or Dexterity increases by 1, to a maximum of 20. Spell Sniper.
</item 13>

<item 14>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Your second most important stat kind of depends on how you play outside of combat, but Constitution is also a good option for the hit points — Warlocks, like
Sorcerers, have a tendency to be glass cannons. Make Charisma your highest stat and Constitution your second highest.

You’ll pick your Warlock subclass at level 1, but the pact you choose here is actually less important than the Eldritch Invocation you get at level 2 — that
said, go with the Great Old One for the most broadly useful features. Your Eldritch Invocation powers up your Eldritch Blast spell — the bread-and-butter spell
for Warlocks — and turns it into an Agonizing Blast.

Best Warlock feats in BG3

Every four levels, you’ll get the option to either increase your stats or choose a Feat. Feats are special talents that add features to your character. If
you’re happy with your stats and start taking Feats, Warlocks can benefit from:

Actor .
</item 14>

<item 15>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Finding the best class really just boils down to how you want to play. Even when playing solo, you’ll still manage the rest of your party, so you’ll get to
experiment with multiple classes at once.

BG3 classes have strengths outside of combat

Your character’s class is mostly about combat, but the underlying Ability Score that it relies on affects the other skills you’ll use to interact with the
world. Baldur’s Gate 3 assigns your Ability Scores automatically (though you can tweak it in the character creator), so each class’s primary ability will be
your highest score.

Bards and Sorcerers are both Charisma-based classes, for example. Their high Charisma will also make them good at skills like Persuasion, Deception, and
Intimidation. Out of combat, those classes will be better at talking their way out of (and into) trouble and lying. Wisdom-based classes like Clerics or Druids
will be more insightful and perceptive of their surroundings.
</item 15>

<item 16>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Should you play a custom or an Origin character in BG3?

Baldur’s Gate 3 gives you the option to play as an Origin character, meaning you can just choose one of the NPCs from your party and play as them right out of
the gate. It’s a perfectly fine way to play if you really don’t want to go through the process of picking out a race and background for your character.

That said, you’ll also be limited to that character’s predetermined story. It’s a bit more work at the beginning, but you’ll have a lot more freedom to
experience the entire story of Baldur’s Gate 3 if you make a custom character. With all that in mind...

Which BG3 class (and subclass) is best for you?

Distinctions like caster vs. puncher or melee vs. ranged don’t really work for Baldur’s Gate 3 (or D&D). There’s a lot of overlap in the classes — especially as
you level up.
</item 16>

<item 17>
How to choose the best class for you in Baldur’s Gate 3 (Polygon, 2023-12-18T17:14:20+00:00)
Most of their magic is nature- or animal-themed and they get Skills to match. What they lack in armor proficiencies — they only get Light and Medium Armor —
they make up for with the ability to shapeshift into a badger, wolf, spider, or cat.

At level 3, Druids pick from the Circle of the Land, Circle of the Moon, and Circle of Spores subclasses. Circle of Land gets you some extra magic and the
ability to cast more spells throughout the day, Circle of the Moon adds a bear to the repertoire of animal shapes you can shift into, and Circle of Spores adds
in some extra fungus- and decay-based magic.

Circle of the Moon is the best subclass for a Druid. Consult our Druid build guide for more details on why, and to see which skills and feats to pick.

Best class for stealth: Rogue

Rogues are Dexterity-based, sneaky fighters. They only get Light Armor, so they have to focus on staying unseen and quick.
</item 17>

<item 18>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
Warlock class features

Hit Points

Hit Dice — 1d8 per warlock level

1d8 per warlock level Hit Points at 1st Level — 8 + your Constitution modifier

— 8 + your Constitution modifier Hit Points at higher levels — 5 + your Constitution modifier per warlock level after 1st

Cantrips

The Warlock starts with two cantrips from the Warlock cantrip list.

Baldur’s Gate 3 Warlock Cantrips Cantrip Description Cantrip Description Blade Ward Take only half the damage from Bludgeoning, Piercing, and Slashing attacks
Bone Chill Prevent the target from healing until your next turn. An undead target receives Disadvantage on Attack Rolls. Eldritch Blast Conjure 1 beam(s) of
crackling energy. Friends Gain Advantage on Charisma Checks against a non-hostile creature. Mage Hand Create a spectral hand that can manipulate and interact
with objects. Minor Illusion Create an illusion that compels nearby creatures to investigate.
</item 18>

<item 19>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
For Warlocks, it’s best to focus on weapon and armor proficiencies:

Drow ’s Drow Weapon Training adds weapons like the rapiers and hand crossbows

’s Drow Weapon Training adds weapons like the rapiers and hand crossbows Humans ’ and Half-Elves ’ Civil Militia add spears, pikes, halberds, and glaives along
with shields

’ and ’ Civil Militia add spears, pikes, halberds, and glaives along with shields Githyanki Martial Prodigy means you get medium armor along with longswords and
greatswords

Martial Prodigy means you get medium armor along with longswords and greatswords Dwarven Combat Training gets you battleaxes, handaxes, light hammers, and
warhammers. The Shield Dwarf subrace also gets you Medium Armor

When you’re creating and leveling up your Warlock, Charisma (for spellcasting) is your most important stat.
</item 19>

<item 20>
The best Warlock subclass, feats, and build in Baldur’s Gate 3 (Polygon, 2023-12-01T22:53:00+00:00)
If you’re bound to The Archfey, you’ll gain an action named “Fey Presence,” which will charm or frighten nearby foes.

The Archfey also has two exclusive spells — Faerie Fire and Sleep.

Best Warlock subclass and build in BG3

Baldur’s Gate 3 races don’t really have any direct impact on your class, but certain races and subraces come with features that play well with certain classes
like extra movement or proficiency with certain weapons.
</item 20>
=== END ITEM id=f3729206d6abc0d4 ===
