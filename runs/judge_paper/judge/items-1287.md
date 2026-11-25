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

=== ITEM id=c7556df20b048c60 ===
Gold sentences and facts:
Sentence 0: In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and effectively.
  Fact 0.0: Scientists point to a number of recent developments in the long term.
  Fact 0.1: The recent developments have raised hopes that dementia may be possible to tackle more directly.
  Fact 0.2: The recent developments have raised hopes that dementia may be possible to tackle more effectively.
Sentence 1: Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies.
  Fact 1.0: Health coach Susan Saunders found herself parenting her mother.
  Fact 1.1: Susan Saunders's mother had dementia.
  Fact 1.2: Susan Saunders's own daughters were babies at the time.

Retrieved items:
<item 1>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
And Sherri Nordlie Geerts was her mother.

Bazella emailed Geerts and they started corresponding regularly.

"I lost 39 years … There’s a lot of grieving with that." — Diane Bazella

"We were slowly getting to know each other through email," Bazella said. "She had a lot on her plate as a caregiver to her husband, so we were slowly
establishing our mother-daughter relationship."

A LIFE IN PICTURES: FROM BIRTH INTO ADULTHOOD, A FATHER TAKES HIS SON’S PHOTO EVERY DAY FOR DECADES

Finally, on Christmas Day 2021, Bazella and Geerts heard each other’s voices over the phone for the first time.

They met in person in July 2022.

Since then, they have been talking on the phone frequently, sometimes for hours at a time.

‘Same sense of humor’

During their conversations, Geerts opened up about having a baby when she was just 18 years old.
</item 1>

<item 2>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
There is no love lost for her family

Britney writes about how she was always made to feel she was not enough as a kid; her father’s alleged alcoholism, neglect and brutal standards; how her mother
always made her feel less-than. She is unsparing about her younger sister, who she characterises as a brat who was spared the pain of Britney’s impoverished
childhood because of her success. “My mom and I had to witness the ugliness and the violence without believing that there was anywhere else to go.”

Even once Britney is the biggest pop star in the world, her parents still seem to have an undue say in the running of her career, including making her sit for
humiliating TV interviews – why exactly that is remains unanswered. Her father seizes control legally with the imposition of the conservatorship, telling her,
chillingly, “I am Britney Spears now,” and her family appear to live off her dime while she lives an extraordinarily restricted life.
</item 2>

<item 3>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 3>

<item 4>
Dangerous 'superbugs' are a growing threat, and antibiotics can't stop their rise. What can? (Live Science: The Most Interesting Articles,
2023-10-01T11:00:46+00:00)
Some lysins have entered mid- and late-stage human trials with hundreds of participants, in which they've been tested as supplementary treatments to antibiotics
but garnered mixed results.

Antibiotic stewardship can save lives, in the meantime

Until these next-gen bacteria slayers make it to market, immediate measures must be taken to stall the rise of superbugs, by preventing the misuse of
antibiotics that pressures bacteria to evolve resistance in the first place.

"By reducing individual risk, you anticipate that you will drop the overall population-level risk." Dr. Shruti Gohil, INSPIRE-ASP Trials

For example, doctors can be more diligent about confirming that bacteria, not viruses, are behind a patient's infection before prescribing antibiotics, said Dr.
Shruti Gohil , a lead investigator of four INSPIRE-ASP Trials , federally funded research aimed at improving hospitals' antibiotic use.
</item 4>

<item 5>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”
</item 5>

<item 6>
Looking for a book to read? Here are 14 tips for October (The Sydney Morning Herald, 2023-10-01T21:04:20+00:00)
His latest brings back Nell Buchanan and Ivan Lucie, whom we met in Treasure & Dirt and Tilt, for a case of murder involving one of the seven families who rule
the Yuwonderie district like a fiefdom. You can expect dark secrets to emerge and herrings to be red.

Life as We Knew it

Aisha Dow & Melissa Cunningham, Scribe, $35. October 3 This account of how the pandemic affected Australia is a timely publication coming only a few days after
Daniel Andrews, the man who approved probably the world’s longest cumulative lockdown, left office. While many people may not want to revisit those stressful
years, Aisha Dow and Melissa Cunningham, journalists with Nine newspapers, manage to make this a riveting story of discovery, dismay, and recovery. Ahead of the
government’s impending inquiry, you might find some answers here.
</item 6>

<item 7>
Author Melissa Lucashenko on playing with black and white binaries (The Age, 2023-10-06T04:50:00+00:00)
Loading “I just had this vague awareness that I should be living in Brisbane to write this book,” she says now, firmly in Brisbane, which is verdant and
infested with aforementioned noisy wildlife. “Because it’s very different to write a book when you’re walking the streets every day and looking at the river
every week, than trying to do it at a distance.” She’s dressed casually in a T-shirt and trousers, finished off with borrowed thongs (her shoes were stolen from
her house this morning). Her eyes are brown, expressive and kind, but she’s wary of too many personal questions.

Advertisement

Later, I look up Reminiscences, which I’ve never read. It was written by Thomas’ daughter Constance Petrie, who states of her father that “no one now living
knows more from personal experience of the ways and habits of the Queensland aborigines. [His] experiences amongst these fast-dying-out people are unique, and
the reminiscences of his early life in this colony should be recorded.”
</item 7>

<item 8>
Looking for a book to read? Here are 14 tips for October (The Sydney Morning Herald, 2023-10-01T21:04:20+00:00)
Particularly family, and particularly her mother, who took her own life at 46 and about whom Davidson has been unable previously to write to her own
satisfaction. She is acutely aware that her version of the truth “can bury another’s”, but this one is hers.

Stone Yard Devotional

Charlotte Wood, Allen & Unwin, $32.99. October 3 Another reckoning with a mother’s death, this time in a novel from the sure hands of Charlotte Wood. A woman
arrives at some sort of religious community having turned her back on her husband, their life and the metropolis. What is she fleeing and what are her demons?
It’s not all peace and tranquillity in the community, which has collective crises to deal with, while her previous life won’t leave her alone. Written in a
spare prose, the consistently brilliant Wood delivers yet again. Will Murdoch help Trump again?
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>

<item 11>
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
</item 11>

<item 12>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
He later issued an apology to Spears on Instagram, writing: “I’ve seen the messages, tags, comments, and concerns and I want to respond. I am deeply sorry for
the times in my life where my actions contributed to the problem, where I spoke out of turn, or did not speak up for what was right. I understand that I fell
short in these moments and in many others and benefited from a system that condones misogyny and racism.”

In June 2021, he also shared his support for Spears after she testified publicly in a bid to end her conservatorship, which was controlled by her father, Jamie
Spears. In a Los Angeles court hearing at the time, Spears alleged that she had been forced to take lithium against her will. She also claimed that the team
managing her conservatorship had prevented her from having another baby by refusing to remove her IUD.

“After what we saw today, we should all be supporting Britney at this time.
</item 12>

<item 13>
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
</item 13>

<item 14>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
They would follow each other in groups and ceased going round in circles. There were some other behaviors, such as attention to a specific task and sleeping,
that were unaffected, but overall the results were promising and according to Yoshizawa, suggest dopamine could be key to how the diet affects behavior.

“We suspect that due to this behavior variation, probably the target of the ketogenic diet is the dopaminergic system,” he said. The challenge now is teasing
out all the complex factors that may be involved.

Diet, environment, and genes – a complex recipe

According to Yoshizawa, there are two plausible ideas as to how the ketones produced by a ketogenic diet are acting on behavior. The first involves the
mitochondria, which use either carbs or fat to produce energy in our cells, and the other involves epigenetics, which simple refers to any non-genetic influence
which turns genes on and off.

Ketones are known to create detectable increases in gene expression in cells.
</item 14>

<item 15>
Shayda director Noora Niasari on family violence, Iranian liberation and taking her first film to the Oscars (The Guardian, 2023-10-01T14:00:47+00:00)
Zar Amir Ebrahimi and Selina Zahednia as Shayda and Mona. Photograph: Miff

It’s a tender and revealing film that balances Shayda’s discovery of inner strength with the sacrifices she makes for her daughter, as she tries to create a new
family for her. It’s understated, relatable and drawn from such personal memories that Niasara describes working on it as “long-term exposure therapy”. Even
doing interviews to promote the movie is difficult. “I have to sit with it and process it,” she says.

“But the thing is, now that it’s a film, it has a really different energy in the world. People bring their own experiences to it, it’s a very universal
experience. We’ve screened it in Europe, North America and Australia and there is a real sense that it connects beyond my mother and I, beyond our experience.
It’s not about us any more. That feels liberating and cathartic.”
</item 15>

<item 16>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
While it was wonderful to have the emotional and practical support, after three months I felt that If I didn’t get out there to live as an independent person, I
would quickly lose any sense of what I was capable of. But I’m incredibly grateful for the time I had living with my parents; I know it’s a huge privilege to
have them as a safety net. I ended up writing a memoir, Inconceivable, about my journey to become a solo mum – and the complexities of being a woman in your
late 30s who has to start over and move back in with your parents. I think while we’re all embracing this contemporary feminist movement where women can create
families of their own making, we’re also realising that we need help, that parenthood can’t happen alone.”
</item 16>

<item 17>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 17>

<item 18>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She suffers from postnatal depression and feels hopeless about her ability to protect her sons: “I got a little depressed once I was no longer keeping them safe
inside my body … I wanted them back inside me so the world couldn’t get at them.”

Every part of normal life had been stripped from me. I had no freedom and yet also no security

She has a keen sense of injustice regarding a perceived contract that she never signed with the public and the press. “They just kept acting like I owed it to
them to let the men who kept trying to catch me looking fat take photos of my infant sons.” Her postpartum body and shots of her without makeup were treated as
“some kind of a sin – as if gaining weight was something unkind I’d done to them personally, a betrayal. At what point did I promise to stay 17 for the rest of
my life?”
</item 18>

<item 19>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 19>

<item 20>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.

On a steamy August morning, Kalmus found himself huddled around a faded picnic table at the center of the homestead in Belfast, Maine, sipping his coffee with a
rare breed — people as alarmed by climate change as him.

climate scientist environmental (Evan Bush / NBC News)

Ethan, the charismatic, folksy co-founder of the Possibility Alliance, started the day off with a poem about a deer in the forest and then began to outline
daily chores — chopping wood, building a new deck, caring for goats.

Soon, talk turned to climate change and Kalmus’ grief began to flow as he shared findings from unpublished research on his projections of heat in moderate
emissions scenarios through the year 2300.

“Even in the coolest time of day, people will be dying if they don’t have air conditioning,” Kalmus told the group.

“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.
</item 20>

<item 21>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
Commercial influence

While many of the questions directed at her over the 1.5-hour long hearing focused on the controversy that’s sprung up around the ad campaign, parliamentarians
also pressed the commissioner on a number of other issues — including concerns about the extent of commercial lobbying around the CSAM-scanning proposal.

This has been a topic of intense interest, especially following a report by investigative journalists published last month by BalkanInsight which looked at
close contacts between Johansson’s department and companies with CSAM-scanning and other child safety tools to sell.

One of the journalists involved in that investigation, Apostolis Fotiadis, had also been invited by the committee to participate in the exchange of views — and
he took the opportunity to defend their reporting from direct public attacks by Johansson.
</item 21>

<item 22>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
At what point did I promise to stay 17 for the rest of my life?”

What she knows now, she writes, “is that every part of normal life had been stripped from me – going out in public without becoming a headline, making normal
mistakes as a new mother of two babies, feeling like I could trust the people around me. I had no freedom and yet also no security. At the time I was also
suffering, I now know, from severe postpartum depression.” It left her suicidal. “I’ll admit it, I felt that I couldn’t live if things didn’t get better.”

She knows she is trapped in a state of arrested development

It’s often said that famous people are frozen in time at the age they became famous, but Britney was never allowed to be a real child, nor a real adult. Her
2001 single I’m Not a Girl, Not Yet a Woman may be her most prescient recording; she is painfully conscious of this divide.
</item 22>

<item 23>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 23>

<item 24>
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
</item 24>

<item 25>
Shayda director Noora Niasari on family violence, Iranian liberation and taking her first film to the Oscars (The Guardian, 2023-10-01T14:00:47+00:00)
It’s not about us any more. That feels liberating and cathartic.”

skip past newsletter promotion Sign up to Saved for Later Free newsletter Catch up on the fun stuff with Guardian Australia's culture and lifestyle rundown of
pop culture, trends and tips Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

The world is seeing the strength of Iranian women now Noora Niasari

After screenings, adult men have approached her with their own tales of growing up in shelters; other audiences have said they left with a deeper understanding
of different Iranian women’s experiences.

It was important to Niasari to shade in light along with the darkness of family violence.
</item 25>

<item 26>
Their kids died after buying drugs on Snapchat. Now the parents are suing (The Guardian, 2023-10-18T10:00:21+00:00)
It forces her to relive horrifying moments again and again. “I don’t know what is worse, getting out and talking about it, or not talking about it,” she said.

But for her, as it does for many parents, suing Snap represents just one iron in the fire. While Mendoza works to spread awareness of the risks of fentanyl to
Spanish-speaking families, Neville travels to schools to share Alexander’s story and hosts monthly online meetings that empower young people to do peer-to-peer
youth outreach.

In doing so, she has learned that many parents and young people alike don’t know about the existence of counterfeit pills – something she desperately wishes she
had been aware of when her son was still alive. The night he died, Alexander had told his parents that he had been taking Oxycontin he got online, and that he
wanted help. Neville and her husband immediately called a rehab facility and made plans to take him there the following day, but didn’t think to take the pills
away.

“We thought we were doing everything right,” Neville said.
</item 26>

<item 27>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
It isn’t ideal, but it’s one step back towards independence. If we’d known then what we know now, we’d never have sold our home.” “It’s a shame we don’t value
multi-generational living more than we do”

Job insecurity and financial practicality encouraged Rachel Claire, a 32-year-old photographer, to move back in with her parents in 2019. “The great thing about
living with your parents when you’re an adult yourself is that you don’t take any of it for granted.” “My parents and I have always been close. When I first
moved out of home at the age of 22, my parents would always find reasons to pop by, whether it was just a quick visit after work or to fix things or mow my
lawn. I could say they struggled with becoming empty-nesters, but I think it’s more accurate to say we’ve just always enjoyed each other’s company.
</item 27>

<item 28>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Ray reassured me that I was not crazy. Google results today do feel different from how they felt just five or six years ago for two major reasons. The first was
Google’s response to the disinformation panic around the 2016 election, which involved questioning the notion that the most reliable information could be chosen
by a form of popularity, meaning how many links a site received from other sites. As a result, the algorithm seemed to change its approach to links, especially
when it came to news and sites offering legal, financial, or health advice, and instead paid more attention to what Google came to call E-E-A-T: experience,
expertise, authoritativeness, and trustworthiness.

“E-E-A-T has had a pretty big impact on what types of results you see,” Ray told me. She’s done extensive (and fascinating) research around how certain sites
have fared under these new guidelines: Urban Dictionary, down! Mayo Clinic, up! Some people consider EEAT part of what’s making results better than ever.
</item 28>

<item 29>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Treatment regimens are now not just restricted to glycemic control. Novo Nordisk’s Semaglutide, a leading GLP-1RA regimen, has shown in studies that it can
reduce an adverse cardiovascular event by 20 per cent in overweight patients. In Type-2 diabetes patients, Sema’s study to show improved outcomes in chronic
kidney disease was successful even on interim analysis.

Semaglutide, branded as Wegovy, showed a 17-18 per cent weight loss sustained over 68 weeks’ trial for obese patients treated with Wegovy. This has created a
flutter as Pfizer, Eli Lilly and others have joined the race in the category.

Convenience is another important dimension on which anti-diabetes treatments place a significant weight. Once-a week insulin, Ozempic, has been approved and
once-a-week long-acting basal analogue is in the works.
</item 29>

<item 30>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
“We sold our home to go travelling and now we can’t get back into the market” Yearning for a tree change, Tess Moone, a 31-year-old travel writer, sold her
Perth home only to discover that a suitable rental was almost impossible to find. “Our initial ‘quick stay’ with Dane’s parents has stretched out month after
month.” “My husband Dane and I became home owners at the age of 21. We thought we were doing all the right things; we bought a house 10 minutes from Perth’s
CBD, where I was working. But after the pandemic we began questioning our lifestyle choices. Did we really want to kill ourselves to pay a million-dollar
mortgage in the city when we were leaning towards a tree change in Margaret River? So we sold our home in February and spent six months travelling around Europe
and Japan. Our plan, once we returned to Perth, was to stay at Dane’s parents’ house while we looked for a rental in Margaret River.
</item 30>

<item 31>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
Wilkins continues to talk to customers. The company has an active beta program with 150 users, and he calls the top 10 users weekly to better understand their
usage. He’s discovered people are now using their Keep for other critical items like keys, passports and cash. He believes these additional use cases show users
trust the device.

With Wandzura leading the company and with gobs of customer feedback, Keep Labs started exploring an enterprise offering focused on medication adherence along
with harm reduction.

The company became part of the McKesson Digital Health Network in Canada to provide real-time data about whether patients are taking their medication. Keep Labs
also has a partnership with Savvy Cooperative to give away free Keep devices to patients who are living with chronic conditions. These partnerships provide an
invaluable feedback loop of front-line patients interacting with their devices.

“Let’s make it super easy to deploy to individuals who need support,” Wilkins said.
</item 31>

<item 32>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 32>

<item 33>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Anyone with older parents should be urging them out for a walk five days a week – and when they have a companion, the walk also doubles as a social occasion,”
he explained. “If they can walk for longer than 20 minutes, even better. Our study showed that the risk of depression fell in proportion to the time spent
walking.”

As well as a daily walk, Laird recommends vitamin D and B12 supplements – both of which appear to moderate the chance of depression in older people. “Depression
matters,” he added. “Not just because it reduces quality of life, but because in older people there seems to be a link between depression and dementia which
we’re still unpacking.”

Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies. “I’d advise anyone with a parent in the
early stages of dementia to investigate all the available research trials. Start by joining Dementia Research, a nationwide register that matches volunteers to
research programmes.”
</item 33>

<item 34>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
When would you pay as much for a robot as you would a car? When it achieves the same level of dependability and value you have come to take for granted in the
amazing machines we use to transport us around the world.

What important robotics story/trend isn’t getting enough coverage?

There is a lot of enthusiasm around AI and its potential to change all industries, including robotics. Although it has a clear role and may unlock domains that
have been relatively static for decades, there is a lot more to a good robotic product than 1’s and 0’s. For AI to achieve the physical embodiment we need to
interact with the world around us, we need to track progress in key technologies like computers, perception sensors, power sources and all the other bits that
make up a full robotic system. The recent pivot in automotive towards electrification and Advanced Driver Assistance Systems (ADAS) is quickly transforming a
massive supply chain. Progress in graphics cards, computers and increasingly sophisticated AI-enabled consumer electronics continues to drive value into
adjacent supply chains.
</item 34>

<item 35>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
With my career as a photographer, I was in and out of the country a lot, spending money on rent when I wasn’t often at home – crazy when you’re in an industry
where there’s very little job security. Eventually, my parents said that maybe it was a good idea to move back home for a while so I could come and go with
minimal financial impact while I was establishing myself. I wasn’t so sure initially; I loved my freedom and struggled with the idea of falling back into
traditional parent/child roles in my late 20s. But when I started thinking about the benefits – particularly my parents’ home security systems, which are very
handy when you’ve got expensive gear – I accepted and moved back in.
</item 35>

<item 36>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
With stories in The Age and Sydney Morning Herald, as well as on 60 Minutes, journalists Nick McKenzie and Chris Masters pushed to bring the truth to the
public, navigating divisions in the SAS and eventually the defamation trial that Roberts-Smith brought to court and lost. It’s been a compelling saga, and Dora
Weekley’s documentary should include hitherto unknown details and unheard commentary.

Advertisement

I still haven’t got over the nightmarish sounds of a spinal surgery going very wrong in the first season, but nonetheless Dr Death (December 22) is back for a
second serve of malpractice and monstrous medical egos. This American true crime drama is now an anthology. Every season is a different doctor whose abuse of
privilege posed a criminal risk to their patients. The new season will cover Paolo Macchiarini, an Italian surgeon and researcher who was accused of laboratory
fraud and manipulative behaviour.
</item 36>

<item 37>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.
</item 37>

<item 38>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
“If the idea of my being in that place was to heal, that was not the effect.”

She’s not scared of anything after that experience, she writes, “but it doesn’t make me feel strong; it makes me sad. I shouldn’t be this strong.”

#FreeBritney protesters in 2021. Photograph: Étienne Laurent/EPA

A nurse shows Britney the blossoming fan movement that is spreading awareness of its suspicions that all is not right with her conservatorship. Come June 2021,
she calls 911 to report her father for conservatorship abuse days before a probate court hearing on the arrangement. “My voice had been used for me and against
me so many times that I was afraid nobody would recognise it now if I spoke freely,” she writes. When she is granted her freedom in November 2021, she says she
feels shock, relief, elation, sadness and joy.
</item 38>

<item 39>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It has been a very long, hard road, but finally we have something positive to look at. That is very welcome.”

Alzheimer’s disease is triggered by the buildup of a protein called amyloid in the brain, although symptoms may not appear for decades after this accretion has
started. Scientists have tried for more than 20 years to find ways to stop amyloid from forming these plaques, in the hope this would stop the progression of
the disease.

Lecanemab, which is produced by Japanese pharmaceutical company Eisai, and donanemab, produced by Eli Lilly of the US, are the first medicines to achieve this
aim – though they only slow but do not ultimately halt the disease’s progression.

Graeme Armstrong with wife Trina, who has a rare form of Alzheimer’s.

Both drugs are to be considered for approval in the UK next year.
</item 39>

<item 40>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 40>

<item 41>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Start by joining Dementia Research, a nationwide register that matches volunteers to research programmes.”

Even if your parents don’t require such intensive “parenting”, a simple supplement could help. A recent study from Columbia University involving 2,200 adults
aged 60 and over found that an inexpensive multivitamin formulated specifically for older people improved cognition, memory and focus. In fact, the results were
so marked that the researchers said the supplement could slow age-related memory decline by three years.

But while most of us can probably persuade our parents to take a multivitamin, suggesting they change their diet might be more complicated. Appetites fade as we
age, chewing becomes more difficult and those living alone don’t always feel inclined to cook. The important thing, say both Saunders and Laird, is that older
people consume plenty of protein.

“Older adults should be taking at least 1 to 1.5g of protein per kilogram of body weight,” says Laird.
</item 41>

<item 42>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
“The different tones that you can tune into can enhance chakras within you and help you heal,” he says.

Healing takes many forms.

When Poyer was 12, Louis Dunbar, who has done multiple stints in prison for violent crimes, called him for the first time and stunned him with the revelation he
was his biological father. In subsequent conversations, Dunbar often said he wanted to see him or come to one of his games. He never has.

“I always had this hate and resentment toward him I was holding,” says Poyer, who doesn’t know where Dunbar is. “This experience enabled me to let that go. I
realize I wouldn’t have anything — my daughter and my wife — if not for him. Even if it was just the seed he gave my mom. So someday I’ll meet him, give him a
hug and tell him I’m sorry.”

Rachel has no interest in taking ayahuasca. But she is grateful her husband discovered it.
</item 42>

<item 43>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Dairy, fish or non-processed meat can help maintain muscle strength and function, reducing the risk of frailty as we age.”

In truth, anyone over 50 would do well to follow these simple guidelines: engage with something new every day, take a daily walk of at least 20 minutes,
socialise regularly, take a daily multivitamin for seniors and check the protein content of our meals. Perhaps we should think of it as self-parenting.
</item 43>

<item 44>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
The study authors concluded that digital engagement – whatever the device – protects the brain. These findings reflect earlier studies that discovered that
older people who made active use of technology were less likely to get dementia.

Dr Eamon Laird, researcher in health and ageing at Limerick university, agrees that we should be encouraging older parents to try new things. And the further
out of their comfort zone they feel, the better. “It’s always good to keep the mind active and fresh,” he told me. “New challenges can help build and maintain
new brain connections and can be good for brain and overall health.”

If older people use technology to keep in touch with extended family and friends, that’s a double whammy. When my 84-year-old mother recently joined Instagram,
she not only expanded her grey matter but she also found an effective means of staying in touch with her grandchildren.
</item 44>

<item 45>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Perhaps we should think of it as self-parenting.

Good groups for elders

Age UK has a database of social activities and groups for older folk

The Walking Football Association keeps keen amateur footballers playing long into their 50s and 60s

Action for Elders offers social and exercise programmes

The Centre for Ageing Better provides information on working and living as an older person

u3a, a charity collective, provides local opportunities for people no longer in work to acquire new skills and knowledge

Annabel Streets is the co-author of The Age-Well Project: Easy Ways to a Longer, Healthier, Happier Life (Piatkus) and co-founder of The Age-Well Project
</item 45>

<item 46>
Shayda director Noora Niasari on family violence, Iranian liberation and taking her first film to the Oscars (The Guardian, 2023-10-01T14:00:47+00:00)
“But I first had to ask my mum for her permission and participation, because I had such a blurry memory of that time.”

Niasari asked her mother to write her memoirs, which took six months; that writing formed the basis of the first incarnation of Shayda’s script. Shayda evolved
over time – and it’s not always a direct mirror of what happened to them both – but “it is very emotionally true to our experience”.

Executive produced by Cate Blanchett, Niasari’s movie tells the story of Shayda (Zar Amir Ebrahimi), an Iranian immigrant in Melbourne who leaves her abusive
husband Hossein (Osamah Sami) with her daughter Mona (Selina Zahednia) in tow. Shayda finds refuge in a women’s shelter where the kindly Joyce (Leah Purcell)
protects and guides her through the tough legal process of a custody fight.

Zar Amir Ebrahimi and Selina Zahednia as Shayda and Mona.
</item 46>

<item 47>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Parenting no longer starts and stops with our children. Nor is it confined to those who have children. In a time of unrelenting change and ever-extending life,
most of us will – at some stage – find ourselves “parenting” our own parents.

Indeed, many of us – particularly those who had families later – will find ourselves simultaneously parenting our kids and our parents. In one breath we’ll be
begging our children to swap French fries for vegetables, and in the next breath we’ll be urging our parents to exchange cake for sardines. Little wonder
today’s midlifers are known as the sandwich generation.

The relentless pace of technology – accelerated by the Covid-19 pandemic – has made everything infinitely more complicated, leaving many older (and not-so-old)
people baffled and frustrated.
</item 47>

<item 48>
Shayda director Noora Niasari on family violence, Iranian liberation and taking her first film to the Oscars (The Guardian, 2023-10-01T14:00:47+00:00)
‘The world is seeing the strength of Iranian women now; it’s not just me looking at my mum in admiration,’ says Niasari. Photograph: Miff

“I grew up with all of the beauty of our culture. So I had a natural inclination to include those moments [in the film]. You can feel the joyous moments even
deeper when you have that tension or darkness around it.”

It has been a year since 22-year-old Iranian woman Mahsa Amini died in custody, after being arrested for allegedly not complying with the country’s hijab laws.
Her death led to a huge wave of popular unrest in Iran; Niasari hopes renewed international awareness of Iranian women’s resistance will lead to more stories
being told.

“The world is seeing the strength of Iranian women now; it’s not just me looking at my mum in admiration. Shayda is just one of those women who happened to
leave Iran and make her own sacrifices and her own life.
</item 48>

<item 49>
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
</item 49>

<item 50>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
While it was lovely to be taken care of, we all tend to regress when we’re living with our parents. It’s impossible not to feel infantilised, and there’s a real
tipping point where you start to lean into learnt helplessness. I stayed for four months before I moved out to house-sit a friend’s place. Then I launched
myself into dating to try to find a man to have a family with. After the roller-coaster of dating, I decided to start fertility treatment on my own, and by the
time it began, I’d found a stable job and was renting an apartment. When I got pregnant, I was elated but also terrified about how I would raise a kid on my
own, financially and emotionally. Luckily, after I gave birth, my parents invited me to live with them, so I moved back for the first three months of my son’s
life.
</item 50>

<item 51>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She accedes in order to retain access to her boys, but questions how she can perform at such a high level when she is perceived as “so sick that I couldn’t make
my own decisions”. She makes futile attempts to escape the arrangement; meanwhile her father sends her to rehab and Alcoholics Anonymous because she has been
taking over-the-counter energy supplements. The women at AA inspire her to start trying to take control over her life but she just hits more walls. Her father
won’t let her remove her IUD to start a family with her new boyfriend, later husband, Sam Ashgari (they have divorced since Britney completed the memoir). After
she objects to a new dance move suggested for her Vegas residency, she is sent to a solitary rehabilitation facility for two months, the most upsetting part of
the book.

Was I a bank robber? A wild animal? Why was I treated as though I were about to burn the place down and murder them all?
</item 51>

<item 52>
Subtle as a sledgehammer, a pop icon returns (The Age, 2023-11-30T18:30:00+00:00)
In fact, there’s a palpable sense of mortality on I/O. Gabriel is 73, and he’s mining a seam of material here about taking stock, feeling at one with the world
and the (admittedly, not exactly original) realisation that love is all we need. This reaches its most intense moment in And Still, a song he wrote to deal with
the death of his mother in 2016. It’s a fragile, delicate thing that has an almost Nick Drake transparency to it, as he remembers leaning his head against her
skin and the feeling of her brushing his hair as a young boy. But with all that time spent cooking up these songs, some have been left in the oven too long,
stretching to five, six, seven minutes, and in need of an edit or at least a change in tempo.
</item 52>

<item 53>
Their kids died after buying drugs on Snapchat. Now the parents are suing (The Guardian, 2023-10-18T10:00:21+00:00)
Brooke Badger, 17, died of an overdose at her California home. Photograph: Courtesy

In California, where Brooke lived, fentanyl-related overdose deaths among 15- to 19-year-olds surged by nearly 800% between 2018 and 2021, according to data
from the California Overdose Surveillance Dashboard. Many are young victims poisoned by counterfeit pills that have been pressed to look like legitimate
prescription drugs, but that are laced with fentanyl, an opioid that is deadly even in granular quantities. Typically, those teenagers acquired what they
believed to be Percocet, Xanax or other pharmaceuticals online through social media.

In their grief, victims’ parents are motivated to end this crisis to prevent another family’s suffering while also giving meaning to their loss. Many have
launched awareness campaigns, founded educational programs and advocated for legislative change. And now, some parents are taking to the civil courts, targeting
the tech giants whose platforms facilitated their children’s purchases of pills that killed them.
</item 53>

<item 54>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
ARTIFICIAL INTELLIGENCE HELPS DOCTORS PREDICT PATIENTS’ RISK OF DYING, STUDY FINDS: ‘SENSE OF URGENCY’

It’s important to protect data privacy and security as well, Patel noted, due to the large amounts of information that must be fed to AI models.

While AI's capabilities are "immense and ever-growing," Patel emphasized that AI’s role is to be an ally.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Humans caring for humans is the bedrock of health care — AI strengthens that foundation, but doesn’t replace it," he said.

"AI can offer insights and assist in decision-making, but human touch, intuition and empathy cannot be replicated by algorithms."

CLICK HERE TO GET THE FOX NEWS APP

While AI promises a "new frontier" in health care — Patel calls for a balance of "enthusiasm and caution."
</item 54>

<item 55>
Author Melissa Lucashenko on playing with black and white binaries (The Age, 2023-10-06T04:50:00+00:00)
Eventually the family settled on Lucashenko’s ancestral land, Bundjalung, in the northern coastal area of NSW, ­buying a property and raising horses. But that
idyll ended in 2007, when Lucashenko and her husband divorced. She couldn’t afford to stay on the land and worried about becoming a “bag lady”. Around that
time, her 16-year-old daughter was hospitalised with mental health difficulties and the two of them moved back to Logan City, poor again. As she said in an
address at the 2013 Melbourne Writers’ Festival: “I knew what life was like in a suburb where the majority of people were ordinary, decent Australians, but a
significant minority were prepared to sell their children’s Ritalin in order to fund a heroin habit. Like some kind of weirdly reverse Charles Ryder coming upon
Brideshead, I’d been there before. I knew all about it.”

Advertisement
</item 55>

<item 56>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
While she has found a place with her birth parents, Bazella knows she may never solve the mystery of how she was switched at birth.

"I lost 39 years," Bazella said. "There’s a lot of grieving with that."

Still, Bazella said she believes she's found unconditional love with her mother.

"It's overwhelming for me coming from where I have been," she added.

"I always felt like I was kind of in survival mode and alone. I finally feel at peace with everything, which is really nice."

CLICK HERE TO GET THE FOX NEWS APP

As for Geerts, she said she hopes to make up for lost time.

"One of the main dreams is that we stay healthy and I live long enough, like a hundred years at least — so we can make up some of this time," Geerts said.
</item 56>

<item 57>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
We’ve also worked out our perfect system for sharing the load; they’ll cook dinner for half the week, for example, and we’ll cook the other half, so the
division of labour has been a positive experience. Loading The most challenging aspects? They’re retired, so they’re home a lot. And I work from home, so it’s a
little tougher for me than it is for Dane, who works in an office. I get lots of invitations from my mother-in-law to join her and her friends for lunch or
coffee when I’m supposed to be working, which is probably not the worst thing to complain about. Despite our love for his parents, the anxiety Dane and I have
been feeling has increased as time’s gone by. So much so that we’ve just agreed to move into a friend’s holiday home, sight unseen, even though we’ll have to
move back in with Dane’s parents during the times the friend wants to use it, such as at Christmas.
</item 57>

<item 58>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
During prolonged periods of fasting, cells undergo a process called autophagy, which involves removing and recycling damaged cellular components, Antoun said.

CLICK HERE TO GET THE FOX NEWS APP

"This process is essential for maintaining proper cellular function and can help protect the brain from neurodegenerative diseases such as Alzheimer’s," he
explained.

More than six million Americans of all ages have Alzheimer's, and every 67 seconds, someone in America develops the disease, according to the Alzheimer’s
Association.
</item 58>

<item 59>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
During an exchange of views with the European Parliament’s civil rights, justice and home affairs (LIBE) committee this afternoon, Johansson admitted the EU’s
executive is investigating the matter.

Initially she had sought to dismiss criticism over the legality of the microtargeting — claiming in a tweet earlier this month (embedded below) that the
campaign was “100%” legal.

Asked by the LIBE committee about the discrepancy between her tweet and the existence of an investigation Johansson said she had been given “new information”
related to DSA compliance that merited looking into.

As my services have been directly accused of illegal acts👇 I think it is import I step in:

1. @EUHomeAffairs have followed the guidelines & the law 100%

2. The promotion of our proposal is standard normal practice

3.
</item 59>

<item 60>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
Key benefits of AI in health care

Patel said he views AI as a "powerful instrument" that helps magnify physicians’ capabilities rather than replacing them.

"One of AI's primary strengths is its ability to recognize patterns that might escape the human eye," he said.

"Whether it's intricate anomalies in medical imaging or subtle patterns in patient histories, AI can pinpoint details that make a significant difference in
diagnosis and treatment."

The technology also helps providers by sifting through and analyzing vast volumes of information, far more than what would be humanly possible, Patel said.

It also automates mundane and routine tasks, allowing health care professionals to redirect their focus to the patient, he noted.

AI TECH AIMS TO HELP PATIENTS CATCH DISEASE EARLY, EVEN ‘REVERSE THEIR BIOLOGICAL AGE’

Early risk detection is another key benefit.

"AI tools can predict potential health risks by analyzing a combination of genetic, behavioral and environmental factors, facilitating early interventions and
potentially saving lives," Patel said.
</item 60>

<item 61>
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
</item 61>

<item 62>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
Williams reportedly appeared tired while on set and didn't seem to be his usual self before he died by suicide on Aug. 11, 2014.

The comedian was diagnosed with the neurological disorder posthumously after an autopsy was performed. Lewy body dementia is the second most common type of
dementia after Alzheimer's. The disease affects brain regions involved in thinking, memory and movement, according to the Mayo Clinic.

The autopsy found no illegal drugs or traces of alcohol in Williams' system.

"The last time I saw Robin Williams, I hadn’t seen him for a few months, and he was thin, and he didn’t recognize my wife," Steven Pearl, a comedian and friend,
said during the documentary. "He hardly said a word. It took him a second to recognize me. I go, ‘Whoa, something’s wrong here.’ I didn't know what. Nobody
would really tell me. And then I never saw him again after that."

CLICK HERE TO GET THE FOX NEWS APP
</item 62>

<item 63>
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
</item 63>

<item 64>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
However, with a “traditional” approval for Eisai-partnered lecanemab for Alzheimer’s (compared to accelerated approval for adacanumab), Biogen has the first
approved treatment, putting adacanumab on the backburner.

Cognitive function improvement is a critical marker for Alzheimer’s treatment and lecanemab delivers lower degeneration compared to placebo on this count. The
treatment reduced clinical decline by 27 per cent at 18 months on clinical dementia scoring and on daily living scale delivered a 37 per cent improvement
compared to placebo. As is evident, the development is satisfactory but a lot more progress is left to be made on Alzheimer’s treatment curve.

Takeaways for Investors

Analysing pharma stocks in the US rests on three factors: drugs about to expire, drugs in ramping up stage and the pipeline.
</item 64>

<item 65>
Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' (Fortune, 2023-10-13T22:26:03+00:00)
They have the money and the power. But we’re getting there, getting more and more women investors, particularly in soccer. Women’s sports, we’ve all been
fighting for it.

What would you like to see in the future for women’s sports?

More. And make sure we get girls early in life into sports. It’s really about the health issue, more than anything. More jobs, more everything. Women of color
and diversity is really important.

We only get 5% of the media. That’s where the money is. People always say, ‘why doesn’t the WTA have as much money as the ATP?’ I’m like, really? If you watch a
show at night, a sports show, just count how many minutes are on men and how many minutes are on women. We’re at 5%. We’ve got to change that.
</item 65>

<item 66>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Timberlake, she likens herself to Benjamin Button. “Somehow that year, in becoming more vulnerable I started to feel like a child again.”

The situation worsens amid postnatal depression. “It was as if some part of me became the baby,” she writes in an apologetic section about transferring her
frustrations on to her interior decorator. “One part of me was a very demanding grown woman yelling about white marble, while another part of me was suddenly
very childlike.”

When she is placed under the conservatorship, she not only regresses, but feels dehumanised by the constant scrutiny. “I became a robot. But not just a robot –
a sort of child-robot. The conservatorship stripped me of my womanhood, made me into a child.”
</item 66>

<item 67>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ketones are known to create detectable increases in gene expression in cells. Pulling apart exactly how things like, diet, environment, genes and
neurotransmitters are linked is incredibly difficult but could reveal which pathways are best to target for autism treatments or could identify a specific
ketone which works more efficiently than others.

While the cavefish are not an exact model for human autism this work shows how these experiments shine light on the biology and evolution of these behavioral
conditions and offer new directions for researchers to follow.

Reference: Motoko Iwashita, et al. Metabolic shift toward ketosis in asocial cavefish increases social-like affinity, BMC Biology (2023), DOI:
10.1186/s12915-023-01725-9

Feature image credit: Masato Yoshizawa

Correction: This article was edited on November 2, 2023. Ryan Lee was originally described as working at the University of Hawai‘i Mānoa.
</item 67>

<item 68>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
It took 10 years for Mike Mackenzie-Smith to get a diagnosis of young-onset Alzheimer’s.

Another grim illustration of the problem facing patients is provided by Graeme Armstrong. His wife, Trina, began having trouble recognising faces and reading
telephone numbers in 2006 and was told, after a CT scan three years later, that she had probably had a stroke – although her symptoms did not suggest such a
diagnosis. It took another three years before she was diagnosed with posterior cortical atrophy, a rare variant of Alzheimer’s disease that affects the way the
brain interprets information from the eyes.

“Had we had an accurate diagnosis four years earlier, Trina could have been placed on the correct medication, which may have been more effective and helped her
day to day,” Armstrong said.

One solution sought by doctors involves the creation of blood tests which could pinpoint the disease quickly and effectively. “These are under research, but we
are several years away from having them in widespread use,” said Thomas.
</item 68>

<item 69>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 69>

<item 70>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Antibodies deal with viruses but fail to do so in HIV infection. By engineering bNAbs in a manner similar to mABs for cancer or vaccines (where antibody
reaction to neutered virus generates immune protection), HIV cure may be on the horizon.

Gilead’s several Phase-2 programmes with ART combination are in progress. Studies as recently as February 2023 presented proof of concept — 90 per cent
participants showed virological suppression at 26 weeks. Incidentally, Gilead’s Sovaldi and Harvoni were the first cures for Hepatitis C. The World Health
Organization (WHO) aims to eradicate the virus by 2030.

Alzheimer’s

The degenerative disease of the brain has proved to be a difficult challenge for pharmaceutical companies. Leqembi (lecanemab), developed by Eisai in
collaboration with Biogen, was the first and only approved drug for Alzheimer’s in July 2023.
</item 70>

<item 71>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
We figured we’d be with my in-laws for three to four weeks, but we didn’t realise how bad the rental crisis in the region is. We’ve been offering more than the
asking price and/or offering more rent upfront to secure a place, but as there’s a preference to rent to families or those who’ve rented before, we keep getting
knocked back. Our initial “quick stay” with Dane’s parents has stretched out month after month. We’re living in one of their spare rooms and have our boxes
stored in another.

I love my parents-in-law, so there have certainly been upsides. We’re spending a lot of quality time together and they’re really lovely about trying to give us
as much space as possible, often booking weekends away so that Dane and I get the house to ourselves for a couple of days.
</item 71>

<item 72>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
With frequent insulin injections being a primary complaint in diabetes treatment regimen, Novo Nordisk has also launched a pill version of its GLP-1RA,
Rybelsus.

HIV

From being a near-fatal prognosis, HIV is now a chronic condition with near-normal life expectancy. The first antiretroviral therapy (ART) started with
azidothymidine in the late 90s, which belongs to the nucleoside reverse transcriptase inhibitor (NRTI) class of medications. The current regimen is a
combination of NRTIs, nonnucleoside NRTIs (NNRTI), protease and integrase inhibitors and casts a heavy pill burden, apart from side effects. ART therapy is
focused on limiting viral replication at several levels of cell development. NNRTI and NRTI act at the cellular level, protease inhibitors work at the
maturation phase, integrase inhibitors at the integration phase of the viral lifecycle.
</item 72>

<item 73>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
While biosimilar launches will impact Keytruda’s sales in 2028, the rate of decline may be lower owing to strong physician preference for Keytruda.

Another important aspect to consider is the pipeline valuation. Here’s an example of how it is done. For instance, Biogen (and Eisai’s) Leqembi for Alzheimer’s
(AD) is expected to generate peak sales of $7 billion by 2030. It has announced a list price of $26,500 for annual course and we assumed 50 per cent rebate to
insurers on average here.
</item 73>

<item 74>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
At the time, her representative also confirmed to People: “Yes, she’s single.”

The news of the breakup came asTMZ alleged that Lucado had cheated on Spears, with claims that her father had purchased a video of the infidelity incident. The
publication later alleged that Lucado denied the cheating allegations, with claims that he was seeing the other woman while he and Spears were broken up.

Despite their seemingly rocky history, Lucado showed his support for his ex in a 2016 interview with theNew York Times about her conservatorship. Speaking to
the publication, he said: “If anyone knew the real Britney, they would know that she would rather be remembered for being the great mother she is rather than
the artist she is.”

The publication also noted that Lucado denied the infidelity claims from Spears’ father, before he added: “And if anyone could see her interactions with her
kids, they would know that there is no need for a conservatorship over Britney’s personal life.”
</item 74>

<item 75>
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
</item 75>

<item 76>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
The article is well worth a read, to figure out in which circumstances you can expect a decent return on investment on your event-going antics.

Today’s newsletter is going to be a bit more to the point than usual: I’m laid up at home with pneumonia (yes, I really am an 86-year-old grandmother.
Surprise!), so forgive the antibiotics-addled ramblings this week. I hope I’ll be back with non-pharmaceutically-enhanced ramblings next week.

Artificial intelligence: Y’all just can’t get enough

Our most-read stories consistently continue to be about AI. No big surprise, perhaps, the AI hype cycle continues apace. This week there’s been a bunch of
stories about the seedy underbelly of AI, including how humans are part of the problem, continuing to trick AI systems into generating boobs and 9/11 memes. Oh,
humans. We spoke with investors to figure out whether the future of AI has hope for us beyond daft memes. (Spoiler alert: Yes.)
</item 76>

<item 77>
Keep Labs’ smart storage redefines medication management (TechCrunch, 2023-10-04T19:24:25+00:00)
Didn’t take your meds? The Keep knows and can trigger a smartphone notification on your or a companion’s phone.

To the founders, Keep is more than just a secure box; it’s a solution to harm reduction and medication management.

I recently spoke with co-founder, president and chairman Philip Wilkins about the company’s journey from a preorder success story to finding an enterprise
offering by partnering with Canadian healthcare providers.

From cannabis to medicine

Wilkins is an earnest entrepreneur. Spend a few minutes with him, and it’s clear he’s passionate about solving the problem of safeguarding medicine.

This was the second time I spoke with him. The first was back In 2019 when his company won an innovation award at CES 2020. This award was later rescinded
because the device was marketed as a cannabis storage device. In 2019, going into 2020, the trade group hosting CES had a strict but convoluted stance on
cannabis products. Somehow, Keep Labs won an award.
</item 77>

<item 78>
Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' (Fortune, 2023-10-13T22:26:03+00:00)
We need to have a league where the very best players will play. We went to Toronto and I did an opening speech about trailblazers. It was amazing because the
families were crying, the players were crying, they said ‘we’ve never been treated like this, it’s amazing, we feel like pros for the first time.’ There were a
lot of little kids there. Kids are going to have an amazing opportunity that the generations before them never had. All three of their networks had it on. It’s
a religion up there.

How is investment in women’s sports changing?

I’m asking CEOs and everyone now — ‘do you invest as much in women as you do in men?’ Then it usually gets quiet. But I must say it’s better than it used to be.
We’re really lucky to be with this investment group. The male allies we’ve had through the years have made such a difference. They have the money and the power.
</item 78>

<item 79>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
When she returns home from the forced stint in a brutal rehab facility, she finds that her parents have thrown away her childhood things, including a binder
full of her poetry. “I felt an overwhelming sadness. I thought of the pages I’d written through tears. I never wanted to publish them or anything like that, but
they were important to me. And my family had thrown them in the trash, just like they’d thrown me away.” She resolves to start over with a new notebook. “In
that moment I made peace with my family,” she writes. “By which I mean that I realised I never wanted to see them again, and I was at peace with that.”

Britney writes that she is working to feel “more compassion than anger” towards them. “It’s not easy.”
</item 79>

<item 80>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Her life under the conservatorship is a horror story

It feels fitting that The Woman in Me is being released a week before Halloween. No matter how much we already know about the 13 years that Britney lived under
a conservatorship, her recounting of its imposition, life under it and her inability to escape it is nothing short of a horror story. It is appalling, like
something out of a Victorian novel, not the very real experiences of a thirtysomething in the last decade.

When she fears that Federline is going to take away access to her kids, she locks herself in a bathroom with one of her sons. “Before I knew what was happening,
a Swat team in black suits burst through the bathroom door as if I’d hurt someone,” she writes, and she is subject to a forced hospital hold. Soon after, her
mother invites her to her beach house because “the cops are after you”. When she arrives, so does another Swat team.
</item 80>

<item 81>
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
</item 81>

<item 82>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Until now doctors have only been able to prescribe medicines that help patients manage their symptoms, so the arrival of the first drugs that treat the actual
cause of the condition has been welcomed – although experts have warned that their use should be treated with some caution.

After decades of research, these are the first drugs to improve patients’ lives directly. If nothing else, they suggest we’re on the right road to tackling
Alzheimer’s

“The new drugs slow down the development of Alzheimer’s by six months to a year and are useful only for those in the early stages of the condition, so they are
certainly not miracle medicines,” said David Thomas, head of policy at Alzheimer’s Research UK.

“However, after decades of research, they are the first to improve patients’ lives directly, and that is a justifiable cause for excitement. If nothing else,
they suggest we are probably on the right road to tackling Alzheimer’s.”

This point was backed by neurologist Cath Mummery, of the Dementia Research Centre at University College London.
</item 82>

<item 83>
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
</item 83>

<item 84>
Their kids died after buying drugs on Snapchat. Now the parents are suing (The Guardian, 2023-10-18T10:00:21+00:00)
“We thought we were doing everything right,” Neville said. “A little bit of information would have gone a long way in our household.”
</item 84>

<item 85>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
How worried are they about getting it wrong? Turns out, not very. Andrews and McDonald know what good advice looks like, and how to give it. “I think the best
advice is often not a statement – it’s a question,” says Andrews. “It’s listening to what someone’s dilemma is, then posing a question that helps them
re-frame.” As two women who are constantly being asked for advice, what are the questions they ask themselves? “What’s the next thing?” says Andrews, without
hesitation. “I’m constantly wondering, what’s the next show that isn’t out there.”

Advertisement
</item 85>

<item 86>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
2016’s Silence is in conversation with, if not a synthesis of, the ideas explored in The Last Temptation of Christ and Kundun; all three are searching,
anti-commercial religious films Scorsese struggled for years to get made.

In all Scorsese’s late-period films (along with the passel of rock documentaries he’s made during this era, including his two on Bob Dylan and one on George
Harrison), Scorsese is consciously returning to certain pet subjects and periods in his work — not to repeat himself, but to revise and add greater perspective.
There’s a sense that he’s trying to adjust his positions with the clear eye of an older person who has seen more of the world, dropped his pretensions, and lost
his capacity for bullshit. His later films return to his old themes and obsessions, but in the process, they’ve become more contemplative.

New tricks

This era has included some radical departures for Scorsese as well.
</item 86>

<item 87>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Considering the 6 million patients in AD in the US, assuming 50 per cent are mild to moderate (applicable group according to FDA label) and 15 per cent on
treatment by year 7, the drug alone can rake in $12 billion (halved to account for Eisai’s share) by 2030 in cumulative sales for a company whose market cap is
$33 billion trading at 14 times FY24 EPS.
</item 87>

<item 88>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
</item 88>

<item 89>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
As well, AI can help enable the delivery of personalized medicine and proactive preventative care, he noted.

"By analyzing individual genetic makeup combined with lifestyle and environmental factors, treatments can be tailored to the unique needs of each patient,"
Patel said.

"And by predicting potential health issues before they manifest, we can guide patients on preventive measures, fundamentally changing our approach from cure to
prevention."

Potential risks and limitations

AI in health care also presents challenges and limitations , Patel acknowledged.

Because AI is a branch of science, it requires rigorous evaluation before it can be applied by doctors, he said.

AI HEART SCAN AIMS TO CATCH BLOCKAGES YEARS BEFORE SYMPTOMS: ‘UNBELIEVABLE BREAKTHROUGH’

"This ensures that we're not just implementing technology for the sake of innovation, but are truly enhancing patient care in a tangible, evidence-based
manner."

AI models also run the risk of bias, he warned.
</item 89>

<item 90>
Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' (Fortune, 2023-10-13T22:26:03+00:00)
Billie Jean King is still globetrotting in support of more investment and equity in women’s sports.

She attended the Women’s World Cup in Australia, kicked off the player draft for the new women’s professional hockey league in Toronto and is opening an office
in London for a tennis business venture involving the international Billie Jean King Cup.

That’s all in the last three months for King, who turns 80 in November.

“We’re kind of at a tipping point,” King said. “People are actually looking at women’s sports like a great investment.”

She’s part of ownership groups involved with the Los Angeles Dodgers, the NWSL’s Angel City FC and the PWHL hockey league that starts in January.

Her busy schedule is reminiscent of the summer of 1973, when a 29-year-old King established the WTA, won the Wimbledon triple crown in singles, doubles and
mixed doubles, achieved equal pay at the U.S.
</item 90>

<item 91>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
However there are long waiting times – up to two years on average – for appointments at these centres.

In addition, diagnosing Alzheimer’s and other forms of dementia is usually based on pen-and-paper tests, followed by lumbar punctures and brain scans before a
final diagnosis.

About 65% of cases are confirmed this way. The remaining third of dementia cases are never diagnosed. Yet patients can only expect to receive treatments –
including the new drugs – if their condition is identified.

Eleanor Mackenzie-Smith’s father, Mike, has young-onset Alzheimer’s. His first symptoms were spotted 17 years ago, when Eleanor was 11. “However, it took over
10 years from the start of his symptoms and four separate tests between 2009 and 2017 to get my dad’s final diagnosis, when he was 65. It was distressing not
knowing what was happening. Too many families, like mine, have had to watch as dementia takes hold of our loved ones while we are left without a diagnosis and
the support and access to treatments.”
</item 91>

<item 92>
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
</item 92>

<item 93>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Being social, says Laird, is as important as trying new things: “Research shows that those who are more social are also more resilient to chronic conditions and
often have a stronger immune system.”

Anyone with older parents should be urging them out for a walk five days a week Dr Eamon Laird, health and ageing researcher

Depression among the elderly is on the rise and often goes unnoticed. Gerontologists now believe that it is one of the most common psychiatric disorders among
the elderly – with half of all cases going undiagnosed. When elderly parents become depressed (many of them come from a generation not accustomed to talking
about their mental health), it can be difficult to know how to respond. But Laird’s latest report suggests that a daily walk could help. His study of more than
4,000 older adults found that as little as 20 minutes of daily walking could cut the risk of depression by up to 43%.
</item 93>

<item 94>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Previous studies have supported this association between fasting and reduced Alzheimer’s risk.

Last year, a study led by the USC Leonard Davis School of Gerontology in Los Angeles also found that mice that followed a fasting diet showed lower levels of
"plaques and tangles" in the brain, reduced brain inflammation and better performance on cognitive tests.

"Cycles of a fasting-mimicking diet (FMD) had strong effects against Alzheimer’s symptoms in two different types of Alzheimer’s mouse models," the lead author
of that study, USC biogerontologist Dr. Valter Longo, told Fox News Digital.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain."

"In mice, the effects of FMD cycles were very strong and wide, meaning they affected both learning and memory and also Alzheimer’s pathology peptides/proteins
(amyloid and tau)."
</item 94>

<item 95>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 95>

<item 96>
Their kids died after buying drugs on Snapchat. Now the parents are suing (The Guardian, 2023-10-18T10:00:21+00:00)
Hanh Badger was working from home the morning of 17 June 2021. She went to the kitchen to grab a second cup of coffee and noticed her daughter’s bedroom door
was still shut. Badger found Brooke, 17, pale and motionless in bed.

Soon, the sheriff arrived and immediately administered Naloxone, a nasal spray that reverses the effects of an opioid overdose. But Badger, a pharmacist, was
confused. Brooke was a talented student who couldn’t wait to begin college that fall.

“She had the whole world in her hands,” Badger said through tears. “There was no way in my mind that [Brooke] had died of a drug overdose.”

In the ensuing days, Badger’s husband and son were able to gain access to Brooke’s computer and, with it, her Snapchat account.
</item 96>

<item 97>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Why was I treated as though I were about to burn the place down and murder them all?

She is locked up, put on lithium, subject to relentless testing and mandatory therapy, can’t see her kids or dog, can’t bathe or dress in private or shut the
door to her room, has set bed and waking times. For blood tests, “the tech drawing my blood would be flanked by the nurse, a security guard and my assistant.
Was I a cannibal? Was I a bank robber? Was I a wild animal? Why was I treated as though I were about to burn the place down and murder them all?” She wonders if
her family is trying to kill her. Made slow and sclerotic, “I began to feel like I was being ritually tortured,” she writes, feeling estranged from her own
body. “If the idea of my being in that place was to heal, that was not the effect.”
</item 97>

<item 98>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 98>

<item 99>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
This comes after several failed attempts, and one controversial false start in the recent years.

Alzheimer’s disease is characterised by amyloid beta deposits on brain cells leading to shrinking and loss of cognitive abilities. Despite a clear marker of
amyloid plaques and a target area to work on, pharma companies failed to address plaque removal or a statistical improvement in cognitive functioning after
plaque removal, till the recent approval. Pfizer, Eli lilly, Merck and many others faced failure in the space. The amyloid theory itself was in question.

Biogen’s adacanumab had a peculiar start. The company itself stopped development in 2019, but it surprisingly pursued a high-dose study into 2021. Later, FDA
approved adacanumab with an accelerated approval for high-dose version, which faced sharp criticism. Three advisers resigned over the approval of the drug that
could have cost $56,000 per annum to administer.
</item 99>

<item 100>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
"I feel that it helps families know we truly do care about them when we’re assessing them – not just because it’s our job, but because we really want them to be
safe."

"Caring Contacts helps me make meaning from every interaction I have with the kids and families — that’s my ‘why,’ and what keeps me going," added Doescher.

To Schalley's knowledge, Children's Nebraska is the first pediatric emergency department to implement a program like Caring Contacts; but she hopes it will not
be the last.

"We have created a replication kit to provide other hospitals and organizations with information on how they can implement the program," she said, noting that
Children's Nebraska has presented the Caring Contacts concept at conferences and received interest from other facilities.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

Schalley suspects that the "intentionality" and deliberately low-tech communication between social workers and teens is another reason the program has worked so
well.
</item 100>
=== END ITEM id=c7556df20b048c60 ===
