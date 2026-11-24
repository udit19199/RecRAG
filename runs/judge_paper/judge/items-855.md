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

=== ITEM id=8593093aaa564eae ===
Gold sentences and facts:
Sentence 0: She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six years.
  Fact 0.0: The speaker explained this at the time of the 2022 Academy Awards.
  Fact 0.1: Will slapped Chris Rock at the 2022 Academy Awards.
  Fact 0.2: At the time Will slapped Chris Rock, the speaker and Will had been separated for six years.
Sentence 1: Outsiders are now aware that Jada and Will have been separated since 2016, but the timeline of her relationship with Alsina is still fuzzy.
  Fact 1.0: Outsiders are now aware that Jada and Will have been separated since 2016.
  Fact 1.1: The timeline of Jada's relationship with Alsina is still fuzzy.

Retrieved items:
<item 1>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Jason Trawick: 2009 to 2013

(Getty Images)

Spears and Jason Trawick, a Hollywood agent who worked at WME, started dating in 2009 before getting engaged in December 2011. However, they called off their
engagement in January 2013 and announced they were going their separate ways.

"Jason and I have decided to call off our engagement. I’ll always adore him and we will remain great friends," Spears said in a statement sent by her
representative to ABC News.

Trawick also issued a statement at the time, saying: “As this chapter ends for us a new one begins. I love and cherish her and her boys and we will be close
forever."

David Lucado: 2013 to 2014

Two months after ending her engagement with Trawick, Spears started dating David Lucado, who worked at a legal firm. In August 2014, Spears announced their
split, tweeting: “Ahhhh the single life!”
</item 1>

<item 2>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“I feel as though Will has a tremendous love and commitment to Jada, as well as to the marriage and the family unit,” she said. “The pride that he takes in
defending his wife is extraordinary, almost to the detriment of his image, at times.”

Ultimately, the public will never fully know what “heavy-duty work” Will and Jada are doing in their relationship, and according to Winter, it’s not necessarily
any of our business. Still, she believes that their choice to separate is something fluid and outside of the traditional marriage design - an evolving model
that could also come to serve other couples. If Will and Jada have been spending seven years apart as a married couple, why would they decide to go back to
living in the same household?

“They’re in the rough with a machete, carving a new path for themselves,” Winter said. “Why would they say: ‘Hey, let’s go back to the house and just relax?’
</item 2>

<item 3>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
During an interview with Hoda Kotb for Today, which aired on 13 October, the actor confirmed that she and her partner have been living “completely separate
lives” for the past seven years. However, according to Jada, their choice to live in separate homes doesn’t mean that they’ll be getting divorced anytime soon.

“I made a promise that there will never be a reason for us to get a divorce,” she said. “We will work through whatever. And I just haven’t been able to break
that promise.”

When asked why she and Will didn’t “just amicably divorce,” Jada simply replied: “We don’t want to.”

She then explained how her family shaped that decision, as she and the King Richard star share two children - Jaden, 25, and Willow, 22. Will also has a
30-year-old son, Trey, who he shares with ex-wife Sheree Zampino. “We love our family and we love each other.
</item 3>

<item 4>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Outsiders are now aware that Jada and Will have been separated since 2016, but the timeline of her relationship with Alsina is still fuzzy. Indeed, the terms of
Will and Jada’s separation are still unknown to the public and in some cases, having separate relationships can give couples the clarity they need about their
marriage.

“What’s pivotal here is, what was the agreement? Did we agree to separate to work on ourselves for the purpose of coming back together again?” Winter said. “Did
we agree to separate and just be alone for a while and not have that discussion? And in the second case, if you have an affair, sometimes it can bring you back
to the original person. You can get whatever you need to achieve or gain a different perspective, allowing you to decide if you want to come back or leave for
good.”
</item 4>

<item 5>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Longtime Hollywood couple Will Smith and Jada Pinkett Smith sent shockwaves around the world this week when the Red Table Talk host revealed that she and the
actor have been separated since 2016. Both Jada and Will have previously doubled down on their stance against getting a divorce, but her recent reveal about
their lengthy separation poses a new question: When should couples just get a divorce?
</item 5>

<item 6>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“We’ve been doing some really heavy-duty work together,” Jada said. “We just got deep love for each other and we are going to figure out what that looks like
for us.”

The pair have been married since 1997 and share son Jaden, 25, and daughter Willow, 22. The Fresh Prince of Bel-Air star also has a son Trey, 30, whom he shares
with his first wife, Sheree Zampino.

As for her children, Jada praised them for how they’ve helped her grow. “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance,”
she told the outlet.

Jada and Will have often opened up about their relationship over the years, and have had very public ups and downs throughout their marriage. Here’s a timeline
of their relationship.
</item 6>

<item 7>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Will Smith y Jada Pinkett Smith (Getty Images)

According to Lucas, some of the controversy surrounding Will and Jada’s relationship ties back to the 2022 Academy Awards - when the King Richard star jumped
onto the stage and struck Rock across the face after the comedian made a joke about Jada’s shaved head, which is a result of her alopecia. “Keep my wife’s name
out of your f***ing mouth,” Will yelled. Now that it’s been revealed the couple was separated at the time, Lucas questioned Will’s behaviour and word choice
during the awards ceremony.

“That’s a weird reaction to have somebody who you’ve been broken up with for six years, you know? So, were they really broken up or is Jada saying that as an
attention-seeking thing?” he asked.

However, Winter still believes the controversial incident speaks different volumes about the work Will has put into his relationship.
</item 7>

<item 8>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
After years of getting along, some tension began to show in 2022. Spears and Federline both shared negative posts on social media about the other, which they
both deleted. Soon after, Preston and Jayden did a rare interview with ITV to discuss the situation.

While Jayden clarified that there was "no hate" involved towards anyone, he said that "it will take a lot of time and effort" to fix their relationship with
their mother. He clarified, "I 100% think this can be fixed. It's just going to take a lot of time and effort. I just want her to get better mentally. When she
gets better, I really want to see her again."

In a message directly to Spears, he said, "I love you a lot, I hope for the best for you. Maybe one day we can sit down like this and talk again."

LIKE WHAT YOU’RE READING?
</item 8>

<item 9>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
“Listen, weddings are beautiful, but they can be very romanticised,” Jada said. “I feel that was a very real moment for the two of us to look each other in the
eyes, recognise that there would be tough times in this journey and to say to each other: ‘No matter what, we’re going to figure it out and that’s why we don’t
need a prenup, because I’m making a promise that divorce won’t be necessary, that we will figure this out.’”

The Girls Trip star explained how she and Will casually promised not to get a divorce, and how that decision impacted their perspective on on marriage.

“And we made that promise to each other without all of the bridal wedding beauty; it was just sitting on a log in his mother’s backyard, and going: ‘Hey’ -
having to really look at the possibility of us not being together,” she said.

This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
</item 9>

<item 10>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Perhaps one of the reasons why Jada and Will have promised not to get a divorce is because they also agreed not to sign a prenuptial agreement. In an interview
with Parade published on 13 October, Jada explained it was her special connection with Will that encouraged them not to sign a prenup before getting married.

“I feel that was a very real moment for the two of us to look each other in the eyes, recognise that there would be tough times in this journey and to say to
each other: ‘No matter what, we’re going to figure it out and that’s why we don’t need a prenup, because I’m making a promise that divorce won’t be necessary,
that we will figure this out,’” she recalled. “And we made that promise to each other without all of the bridal wedding beauty; it was just sitting on a log in
his mother’s backyard, and going: ‘Hey’ - having to really look at the possibility of us not being together.”
</item 10>

<item 11>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“I told Will from the gate, I said let me tell you something: ‘If you marry me, know this: we’re gonna be together. We’re going to be under the same roof,’” she
said during a 2018 episode of Red Table Talk. “For me personally, I’m not mature enough to have a divorce, I’m just not. I don’t think I would ever be mature
enough, I don’t.”

Will - who divorced his first wife, Sheree Zampino, in 1994 - has also shared similar sentiments about making his marriage with Jada work. Back in 2006, the
Fresh Prince of Bel-Air star first explained to MTV News that divorce wasn’t an option for him and his second wife. “With Jada, I stood up in front of God and
said: ‘Til death do us part,’” he said at the time. “So there are two possible outcomes.
</item 11>

<item 12>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.
</item 12>

<item 13>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 13>

<item 14>
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
</item 14>

<item 15>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
Although they’ve made their dedication to each other quite clear, there have been some grey areas over the last seven years about the terms of Will and Jada’s
separation. In 2020, Jada made headlines when she revealed in a Red Table Talk episode with her husband that she had a relationship with Alsina. She explained
that she and her husband were “separated amicably” when the romance occurred, which had not been disclosed to the press at the time. “We decided that we were
going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,” Jada said. As the Girls
Trip star recalled that she “got into a different kind of entanglement with August,” she was prompted by Will to confirm it “was a relationship, absolutely.”
</item 15>

<item 16>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It may be over a decade since Britney Spears and Kevin Federline ended their marriage, but the former couple continue to have their ups and downs when it comes
to co-parenting their two teenage sons, Sean Preston and Jayden.

Ahead of the release of the pop star’s memoir, The Woman in Me, which is due for release on 24 October, fans are anticipating Spears will share new details
about her relationship with her second husband, Federline.
</item 16>

<item 17>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
However, there are still certain behaviours in a marriage that could signal it’s time to get a divorce.

“If you feel disrespected by your partner, feel humiliated, or were cheated on, you probably feel ready to split,” Lucas said. “I also think a real big one is,
if you lose sexual attraction for your partner and you’ve actively tried to get it back and can’t, it’s time to go.”
</item 17>

<item 18>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
he wrote on Facebook at the time. “So, in the interest of redundant, repetitious, over and over-again-ness… Jada and I are… NOT GETTING A DIVORCE!!!!!!!!!!!!!”

2018: Will revealed that he and Jada don’t refer to themselves as married

In an episode of TIDAL’s Rap Radar podcast, Will said he and Jada “refer to ourselves as life partners, where you get into that space where you realise you are
literally with somebody for the rest of your life. There’s no deal breakers. There’s nothing she could do - ever. Nothing that would break our relationship. She
has my support till death and it feels so good to get to that space.”
</item 18>

<item 19>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“I already read it and I’m very proud of her,” he told the outlet. “She put a lot of work into it, and it was very hard. I’m definitely excited, and I’m
extremely proud of her. It was a very tough one [to write].”

The Woman in Me is due for release on 24 October in the US and UK.
</item 19>

<item 20>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
But the amount of time they’ve spent separated begs questions about all long-term relationships in general. When is it the time for couples to throw the towel
in? And when should separations ultimately become the end result that couples have tried to avoid: divorce.

While there’s never a simple answer to the question, relationship experts have discovered the signs that it’s time to keep fighting for the one you love, or
time to let them go.

Speaking to The Independent, New York City-based relationship expert Susan Winter highlighted the jarring differences between a separation and divorce. When it
comes to divorce, Hollywood couples can notably get involved in lengthy legal processes, from dividing financial assets to creating child custody agreements.
Although we don’t know the terms of Jada and Will’s separation, Winter believes that they’re showcasing a non-conventional relationship that’s not familiar to
the public. Will and Jada’s dynamic includes living in separate homes, but it could be a choice that’s for the better.
</item 20>

<item 21>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.
</item 21>

<item 22>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
The Independent has contacted u/PopShort7064 for comment.
</item 22>

<item 23>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
At the time, her representative also confirmed to People: “Yes, she’s single.”

The news of the breakup came asTMZ alleged that Lucado had cheated on Spears, with claims that her father had purchased a video of the infidelity incident. The
publication later alleged that Lucado denied the cheating allegations, with claims that he was seeing the other woman while he and Spears were broken up.

Despite their seemingly rocky history, Lucado showed his support for his ex in a 2016 interview with theNew York Times about her conservatorship. Speaking to
the publication, he said: “If anyone knew the real Britney, they would know that she would rather be remembered for being the great mother she is rather than
the artist she is.”

The publication also noted that Lucado denied the infidelity claims from Spears’ father, before he added: “And if anyone could see her interactions with her
kids, they would know that there is no need for a conservatorship over Britney’s personal life.”
</item 23>

<item 24>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears is also the mother of two children – Sean, 18, and Jayden, 17 – who she shares with her second ex-husband, Kevin Federline.

According to a press release for The Woman in Me, Spears’ memoir is “a brave and astonishingly moving story about freedom, fame, motherhood, survival, faith,
and hope”. Spears’ book is also expected to cover the high-profile legal battle to end her conservatorship, which was dissolved in November 2021 after 13 years.

Throughout her time in the spotlight, specifically during the ‘90s and 2000s, Spears has been in some heavily publicised romances. Unsurprisingly, fans are
anticipating that Spears will detail some of her relationships in The Woman in Me, which is set for release on 24 October.

From Justin Timberlake to her now ex-husband Sam Asghari, here’s everything we know about Britney Spears’ relationships and rumoured romances.
</item 24>

<item 25>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
He went on to explain that he’s “happier than [he’s] ever been in [his] entire life” and praised Jada for continuing to be by his side.

“I have achieved every single thing I have ever wanted in my entire life. There is not one thing left that I desire in this world - all of the awards, all of
the money, the family - everything I’ve ever dreamed,” he said. “And those dreams were largely built on the foundation of Jada’s sacrifices. There were many
times when Jada put her career aside so I could follow the dreams of mine.”

The Fresh Prince of Bel-Air star thanked his wife for putting him “before herself” and for maintaining a good relationship with his ex-wife Sheree Zampino. Will
and Zampino share a 30-year-old son, Trey Smith, who was also present for the book event at the Enoch Pratt Free Library.
</item 25>

<item 26>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Will and Jada with their son Jayden in 2003 (Getty)

2013: Jada denies open marriage rumours (sort of)

After a quiet decade and a half making movies and raising their children, Jada took to Facebook to address rumours she and Will were in an open relationship.

She wrote: “Here is how I will change my statement... Will and I BOTH can do WHATEVER we want, because we TRUST each other to do so. This does NOT mean we have
an open relationship... this means we have a GROWN one.”

2015: Will addresses divorce rumours

Will addresses persistent separation rumours by clarifying the pair were not getting a divorce.

“Under normal circumstances, I don’t usually respond to foolishness. But, so many people have extended [to] me their ‘deepest condolences’ that I figured -
‘What the hell… I can be foolish, too!’” he wrote on Facebook at the time.
</item 26>

<item 27>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
She has my support till death and it feels so good to get to that space.”

2020: August Alsina claims he had an affair with Jada

August Alsina and Jada were introduced in 2015, but it wasn’t until 2020 that he claimed that he and Jada had dated while she was married and that her and
Will’s marriage was open during that time.

Alsina said Will gave him and Jada his “blessing”, adding: “I actually sat down with Will and had a conversation due to the transformation from their marriage
to life partnership that they’ve spoken on several times and it not involving romanticism.”

Alsina added he “totally gave myself to that relationship for years of my life and I truly and really, really deeply loved and have a ton of love for her.”

The same day Alsina’s claims surfaced Jada’s reps said they were “absolutely not true”.
</item 27>

<item 28>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
I don’t think I would ever be mature enough, I don’t.”

The Fresh Prince of Bel-Air star has also shared similar sentiments, as he said during an appearance on the Ellen Degeneres Show that “divorce just can’t be an
option” for him and Jada. “It’s really that simple. And I think that’s the problem with LA - there are so many options. So a huge part of the success for [Jada]
and I is that we just removed the other options.”

Last week, Jada made headlines when she revealed that at the time of the 2022 Academy Awards, when Will slapped Chris Rock, she and her husband had been
separated for six years. “We’ve been doing some really heavy-duty work together,” she said during an interview with People. “We just got deep love for each
other and we are going to figure out what that looks like for us.”
</item 28>

<item 29>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
</item 29>

<item 30>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
This isn’t the first time that Jada and Will have taken some time apart from each other. During a 2020 episode of her Facebook Watch series, Red Table Talk,
Jada disclosed that while she and Will were separated amicably, she had a relationship with singer August Alsina, describing their involvement as an
“entanglement”.

Throughout their marriage of 26 years, the A-listers have made it clear that divorce was never in the cards. Jada furthered that belief during a preview for an
interview with Today’s Hoda Kotb, which aired on 13 October. “I made a promise that there will never be a reason for us to get a divorce,” she said. “We will
work through, whatever. I just haven’t been able to break that promise.”

Of course, outsiders aren’t fully aware of the “heavy-duty work” that Jada and Will have done throughout their marriage.
</item 30>

<item 31>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“So there are two possible outcomes. One, we are going to be together ‘til death, or two, I am dead.”

Speaking to The Independent, Lucas noted that while it’s important for both parties to be invested in their relationships, promising not to get a divorce could
spark feelings of resentment down the line. Fans aren’t fully aware of why Will and Jada made that promise to each other, but the dating coach believes that an
agreement like theirs could create a false narrative about a divorce.

Jada Pinkett Smith y Will Smith (Getty Images)

“Getting divorced shouldn’t feel like a taboo because people grow apart and that’s fine,” he explained. “People get stuck in relationships because of their
friends and family or societal pressure, and they worry about getting divorced. They think a divorce has a stigma to its name, which it doesn’t, and they push
aside the fact that they still essentially feel trapped with someone they don’t want to be with.”
</item 31>

<item 32>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
His recent comments come after Jada made major revelations about their marriage in her memoir, including how she and Will have been separated for seven years.

During the event in Baltimore, Will began by speaking to the crowd about his love and admiration for his partner, who he married in 1997. “I just really wanted
to come out and just be here and hold it down for you the way you have held it down for me,” he said during the event, as reported by The Baltimore Sun.

The King Richard star went on to describe the challenges and achievements he’s had throughout his marriage, adding: “We have had a very, very long and
tumultuous [relationship]. We call it ‘brutiful.’ It was brutal and beautiful at the same time."

At another point during the event, Will referred to his union with Jada as a “sloppy public experiment in unconditional love,” according to the Baltimore
Banner.
</item 32>

<item 33>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
1997: Will and Jada get engaged and married

Will popped the question to Jada in November 1997 before getting married just one month later in Baltimore.

Jada wasn’t initially sold on the idea of marriage. She told People in 2019: “I never wanted to get married. But my mother was like: ‘You have to get married’ -
she’s so old-school - and Will wanted a family. So I said: ‘All right, maybe it’s something I should do.’”

1998: Their first child is born

Jada and Will’s first child together, son Jayden, was born in July 1998.

2000: Their second child was born

Jada and Will welcomed daughter Willow in 2000.
</item 33>

<item 34>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Speaking to the crowd, Will said that Jada’s positive relationship with Zampino allowed all three of his children the chance to grow up together.

“Can you show up and love somebody for the rest of your life, even when you don’t agree with them?” he asked. “Jada is the best friend I have ever had on this
planet and I am going to show up for her and support her for the rest of my life.”

This isn’t the first time that Will has opened up about his relationship, as it’s continued to make headlines. Earlier this week, he appeared to mock the
publicity storm surrounding him and Jada in a video titled “Official statement” shared on Instagram. “So, here’s the thing. My opinion is... uh…” the actor
said, before sneezing into the crook of his arm. As he sneezed, the video comically zoomed out to show several different landscapes including lakes and
mountains.
</item 34>

<item 35>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“It saddens me to hear that my ex husband has decided to discuss the relationship between me and my children … As we all know, raising teenage boys is never
easy for anyone,” she wrote. “It concerns me the fact that the reason is based on my Instagram … It was LONG before Instagram … I gave them everything … Only
one word: HURTFUL.”

Asghari went on to accuse Federline of using their sons to “vilify” his wife, and claimed her ex-husband’s “gravy train will end soon, which probably explains
the timing of these hurtful statements.”

“There is no validity to his statement regarding the kids distancing themselves and it is irresponsible to make that statement publicly,” Asghari wrote on
Instagram.

Following Federline’s comments, Spears’s attorney Mathew Rosengart said in a statement obtained by Variety that Federline has created “legal issues” for
himself, after posting private videos of Spears sternly speaking with her children without her consent.
</item 35>

<item 36>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.
</item 36>

<item 37>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”
</item 37>

<item 38>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed why she and her husband Will Smith never signed a prenuptial agreement before getting married.

The former talk show host, 52, spoke candidly about her husband during a recent interview with Parade. While discussing her new memoir, Worthy, Jada revealed
that she and Will didn’t sign a prenup when they were married in 1997 because they knew divorce wasn’t in the cards for them.
</item 38>

<item 39>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Here’s a timeline of their relationship.

1994: Will and Jada meet on the set of The Fresh Prince of Bel-Air

Will and Jada met after she auditioned for the role of Will’s girlfriend on The Fresh Prince of Bel-Air. While she did not get the part, the pair hit it off -
despite Will being married to Sheree Zampino at the time, with who he shares son Trey.

“I had a realisation I wasn’t with the person I was supposed to be with,” he told Jada’s Red Table Talk in 2018. “I was sitting in a [bathroom] stall and I was
crying and laughing uncontrollably, and I knew [Jada] was the woman I was supposed to be with.”

Will and Jada in 2000 (Getty)

1995: The pair officially begin dating

It wasn’t until 1995 that Will and Jada began seriously dating after Will finalised his divorce from Zampino.
</item 39>

<item 40>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
That same day, the fitness instructor officially filed for divorce, listing the date of separation as 28 July 2023 and citing "irreconcilable differences" as
the reason for the split.

In a statement posted online, Asghari said: “After six years of love and commitment to each other my wife and I have decided to end our journey together. We
will hold onto the love and respect we have for each other and I wish her the best always. S*** happens. Asking for privacy seems ridiculous so I will just ask
for everyone including the media to be kind and thoughtful.”

Days before Spears’ memoir release date, Asghari revealed that he had obtained an early copy of the book. In an interview with TMZ on 13 October, he explained
that he wasn’t worried about anything being written in the book that could paint him in a negative light.

“I already read it and I’m very proud of her,” he told the outlet.
</item 40>

<item 41>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
While celebrities like Madonna, Paris Hilton, and Selena Gomez attended the festivities, Spears’ close family – her mother Lynne, father Jamie, sister
Jamie-Lynn, and her two sons – were not present at the nuptials.

Although their marriage last only 14 months, with Asghari filing for divorce in August this year, the actor and former dancer was previously praised by fans
over his continuing support for Spears amid her conservatorship and custody battle.

In August 2022, Federline claimed during an interview with ITV that their teenage sons found it “tough” seeing their mother sharing nude photos online, saying:
“I can’t imagine how it feels to be a teenager having to go to high school.”

Spears hit back at her ex-husband in a statement shared to her Instagram Stories responding to Federline’s “hurtful” comments.
</item 41>

<item 42>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
But the structure of the life that we had established was strangling both of us.”

2022: No infidelity statement and Oscars incident

During an interview on CBS Sunday Morning earlier this month, Will said he and Jada had never accused each other of cheating.

He said: “Never. There’s never been infidelity in our marriage...And we have never surprised one another with anything ever..I have decided that chatter about
my life can be of a benefit to people. I think that chatter is the first stage to having a real conversation and being able to truly explore if some of the
things in your heart are loving, or poisonous.”

In March 2022, Will slapped Chris Rock at the 94th annual Oscars before he yelled at the comedian to “keep my wife’s name out your f***ing mouth”. It quickly
became the most talked-about moment of the evening with the pair initially appearing to laugh off the tense moment.

The altercation came after Rock said: “Jada, I love you.
</item 42>

<item 43>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“It’s an LAT partnership, meaning ‘living apart together,’” Winter said. “While everybody is at a different point as to why they are doing this, the centre
point is the same: the preservation of the partnership is paramount. Separations can also be beneficial if both individuals are working on themselves and have
the mutual goal of reuniting, or making their relationship better. However, that doesn’t necessarily mean that their improved relationship will result in them
living together again.”

On the other hand, dating coach Jacob Lucas - who’s based in the UK - acknowledged that not all separations end in resolutions. When couples opt not to live
together anymore, it could create an emotional disconnect between them. If a lack of connection occurred before couples separated, that separation doesn’t
necessarily make the issues go away. In fact, Lucas believes that in these situations, couples could also be avoiding the inevitable breakup for the sake of
comfortability.
</item 43>

<item 44>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
They made their relationship Instagram official in 2017 and went on to support each other through different milestones - including the 29th annual GLAAD Media
Awards, where Spears accepted the Vanguard Award for her work promoting equality and acceptance.

In April 2022, Spears announced she was pregnant with her and Asghari’s first child together, and her third child. However, the following month, she said in a
statement posted to Instagram that she had suffered a miscarriage. The couple wrote: “We have lost our miracle baby.”

That June, they held their lavish nuptials at her home in California with a star-studded guest list, which included Selena Gomez, Paris Hilton, Drew Barrymore,
and Donatella Versace - who designed Spears’ wedding dress.

In August, months after Asghari shut down speculation about getting a divorce, TMZ claimed that he had confronted Spears about rumours she cheated on him.
</item 44>

<item 45>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
“We love our family and we love each other. It’s more of a life partnership," Jada explained. "Now, 10 years from now, Hoda, who knows?”

During a 2018 episode of her Facebook Watch series, Red Table Talk, Jada doubled down on her stance about divorce, as she recalled the the hard conversation she
had with Will about the topic before they got married.

(Getty Images)

“I told Will from the gate, I said let me tell you something: ‘If you marry me, know this: we’re gonna be together. We’re going to be under the same roof,’” she
said. “For me personally, I’m not mature enough to have a divorce, I’m just not. I don’t think I would ever be mature enough, I don’t.”
</item 45>

<item 46>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
However, during an appearance on the Today show on 16 October, Jada clarified to Hoda Kotb that she and the Suicide Squad actor are “working hard” and are
“concentrating on healing” to get their relationship to a better place.

“There’s no finding another great love, and I think that’s the point,” she said. “It’s like we are in a place now that we are in a deep, healing space. And we
are really concentrating on healing the relationship between us.”

The Girls Trip star continued: “There’s no divorce on paper. We really have been working hard. That’s the whole thing. We are working very hard at bringing our
relationship together. Back to a life partnership.”
</item 46>

<item 47>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
They’re really exploring something vastly different. It doesn’t end up that you go back into that little box of ‘this is a marriage, we’re living together’. It
just doesn’t do that.”

Every relationship is different, living in the public eye or not. Although living separately may not be an issue for some couples, there can still be some major
signs that it’s time to leave a relationship behind. “When there is no stone left unturned, or when they have tried every single combination possible to make
the relationship function and yet it doesn’t, that’s the time to pull the plug,” Winter added. “They’ve done their individual work and they have done the work
together and ended up in the same place.”

Will and Jada’s relationship could end in a variety of ways. The celebrity couple could stay separated, move back in together as a married pair, or they can get
a divorce.
</item 47>

<item 48>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.
</item 48>

<item 49>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“It’s hard... the thing Will and I talk about a lot is the journey. We started in this [relationship] at a very young age, 22 years old... I think you expect
your partner to know - especially when it comes to sex. It’s like: ‘Well, if you love me, you should know. If you love me, you should read my mind.’ That’s a
huge pitfall,” she said at the time.

The next month, Will’s memoir is released which details several issues the couple have had over the years. He wrote they were “suffering the brutal death of our
romantic fantasies, the burning away of the idealistic illusion of the perfect marriage and the perfect family. Neither of us wanted a divorce; we knew we loved
each other, and some aspects of our union were magical. But the structure of the life that we had established was strangling both of us.”
</item 49>

<item 50>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
(AFP via Getty Images)

“They’re not resolving their problems, they’re rescheduling them. So, if you separate from someone, but you live in a separate house and you’re still together,
it seems to me you’re just not letting go of the past because you want to stick to what’s familiar,” he claimed. “People stick to what they know, and it’s like
you have a tether to that person. Your brain’s like: ‘I cannot let go of them because I am scared of the unknown.’ I don’t think I’ve ever seen a couple
separate and come back again.”

It’s unclear whether Jada and Will plan on ending their separation and living in the same home again. However, the couple have been candid about their plans to
not get a divorce. Jada previously recalled that when she first decided to get married to Will in 1997, it would be for the long haul.
</item 50>
=== END ITEM id=8593093aaa564eae ===

=== ITEM id=6e41e7ab645cdc2f ===
Gold sentences and facts:
Sentence 0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has been more than just a curiosity.
  Fact 0.0: For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme.
  Fact 0.1: Inside the Big Ten, the alleged Michigan sign-stealing scheme has been more than just a curiosity.
Sentence 1: While it is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough stain to erase knowing those violations are now on the Big Ten record.
  Fact 1.0: It is almost impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons.
  Fact 1.1: The in-person scouting and sign-stealing are alleged violations.
  Fact 1.2: The alleged violations are now on the Big Ten record.
  Fact 1.3: Knowing the alleged violations are on the Big Ten record will be a tough stain to erase.

Retrieved items:
<item 1>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Earlier this week, Central Michigan said it is investigating whether suspended Michigan staffer Connor Stalions was on the CMU sideline during the Chippewas’
Sept. 1 game at Michigan State. Screenshots of a person who looks similar to Stalions began circulating online Monday night, and The Athletic obtained more
photos of the person on the sideline Tuesday.

Central Michigan is investigating if the person on the right here is Connor Stalions on the CMU sideline at Michigan State earlier this season. AD Amy Folan:
“We became aware of these photos late yesterday and we are in the process of determining the facts surrounding them." pic.twitter.com/ncazAghBbS — Chris Vannini
(@ChrisVannini) October 31, 2023

Stalions, who was suspended with pay by Michigan on Oct. 20, is at the center of the NCAA’s investigation into the alleged scouting and sign-stealing scheme.
</item 1>

<item 2>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.



He was emotional after defeating Penn State.



"I want to thank Coach Harbaugh. ... I love the s— out of you man."



🎥: @CFBONFOX pic.twitter.com/NmXZWrGxCM — The Sporting News (@sportingnews) November 11, 2023

Moore, however, will emerge as a leading candidate for some jobs as a result, and that is a huge victory for one of the nation’s top assistants.

Losers

Connor Stalions

Stalions made a name for himself. He will be remembered as the Michigan staffer as the center of this sign-stealing scandal. Stalions resigned on Nov. 3. The
NCAA investigation whether Michigan was involved with in-person scouting, which is prohibited by the NCAA, put Stalions into the college football consciousness.
</item 2>

<item 3>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
ESPN reported that Big Ten coaches and athletic directors encouraged Big Ten commissioner Petitti to act last week, but that resulted in action before the NCAA
investigation concluded.

Do you wonder what the SEC and commissioner Greg Sankey privately think about all this? For the Big Ten, it is an ugly run of bad behavior on and off the field
for a conference that is welcoming four new members next season in Oregon, UCLA, USC and Washington.

To be determined

Michigan

Michigan is 10-0 and on track to meet No. 1 Ohio State for another blockbuster matchup in college football’s best rivalry. On the field, the Wolverines remain
on track for a third straight Big Ten championship and shot at the College Football Playoff.

But this is about more than football.

Will more penalties come from the Big Ten and NCAA as a result of the in-person scouting scandal that has not left the college football consciousness this
month? Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan?
</item 3>

<item 4>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 4>

<item 5>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
1 Ohio State is a huge television draw and Petitti has a background in TV with the MLB Network. Petitti went heavy-handed instead, maybe a little too heavy
considering the pending legal response from Michigan.

This no doubt is risky on several fronts. The Big Ten made it clear that they still have not linked Harbaugh to any wrongdoing and that this an institutional
penalty based on the findings. The NCAA also will continue its investigation into the Wolverines. The timing of Friday's announcement also was poor – less than
24 hours before the game against Penn State. Petitti upheld sportsmanship within the conference, but it could lead to a long-term rift with Michigan.

Regardless, Petitti did the right thing.

DECOURCY: Michigan got off easy with Harbaugh suspension

Ohio State-Michigan TV ratings

With or without Harbaugh, The Game will draw ridiculous television ratings.

According to Sports Media Watch, three college football games have drew a Nielsen rating of 5.0 or better this season.
</item 5>

<item 6>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Helmet technology

CBSSports.com reported on Oct. 20 that the NCAA rules committee is giving teams the option of coach-to-player helmet communication to relay plays from the
sideline during bowl season this year. Tablets will inevitably follow, and here is the sliver of good that will come from this scandal.

It's about damn time. Death to sandwich-board playcalling, please. Bring on those playsheets where coaches simply have to cover their mouth. This is one
instance where emulating the NFL game is overdue.

Sherrone Moore

If Moore is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including
32 straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.

Moore was emotional - complete with NSFW language – in the aftermath.

Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will Harbaugh finally take the parachute to the NFL? Those are questions bound to come up at the end of the 2023 season.

BENDER: Michigan will need Jim Harbaugh against Ohio State

Big Ten

This is the fourth football scandal in less than a calendar year.

Three former Iowa football players and a student manager were charged in an illegal sports wagering scandal. Several student-athletes face allegations of
placing bets on games in which they played in. Northwestern coach Pat Fitzgerald was fired on July 10 after a hazing investigation into the program. Fitzgerald
is suing the university for wrongful termination. Michigan State fired Mel Tucker on Sept. 27 after allegations of sexual misconduct from Brenda Tracy, a
speaker on sexual assault awareness.

Now, Michigan has a scandal on the field, and they countered with allegations that Ohio State and Rutgers leaked Michigan's signs to Purdue in last year's Big
Ten championship game.
</item 7>

<item 8>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
For the past two weeks, coaches across college football have been riveted by the alleged Michigan sign-stealing scheme, but inside the Big Ten, the topic has
been more than just a curiosity. On Wednesday’s Big Ten coaches video call with commissioner Tony Petitti, and after Jim Harbaugh left the call, that
frustration was voiced loud and clear, according to conference coaches, who said they don’t feel like the new Big Ten commissioner is “motivated” to do anything
about the Wolverines.

Advertisement

“There is just a ton of frustration,” a Big Ten coach told The Athletic on Thursday morning. “Look at Jim Harbaugh’s record before this started. The guy was on
the hot seat before 2021, and now he’s like the king of college football. … No doubt this all has had a profound effect.
</item 8>

<item 9>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
We (the Big Ten head coaches) want something done now. I don’t think people understand the advantage that what they’re (allegedly) doing gives you. People
think, ‘OK, now that everybody knows, we all can just move on.’ Like, ‘now, it’s fair.’ Well, no, it isn’t. Not at all. This changes the way you operate. A lot
of teams have been doing things a certain way for years. Now, it’s forcing you to teach your players a whole new way to communicate just for them. People think
that this is just advanced scouting. This was damn near espionage.”

A third Big Ten head coach told The Athletic that this is “one of the most egregious breaches in the spirit of the game” he’s ever heard of.

“They (Michigan) have been manipulating the game and cheating the game for two-and-a-half years.
</item 9>

<item 10>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
The Big Ten announced Michigan coach Jim Harbaugh will be prohibited from coaching on the sideline for the remainder of the regular season.

At least, in theory. The Friday ruling is the next chapter in an NCAA investigation into Michigan for in-person scouting and sign stealing in a scandal that led
to the resignation of staffer Connor Stallions on Nov. 3. Harbaugh will be allowed to coach during the week but will not be able to be on the sideline. Michigan
did not get a ruling on a temporary restraining order ahead of the showdown against No. 10 Penn State.

The Wolverines still won 24-15 under interim coach Sherrone Moore, but the fallout from the scandal continues to have a far-reaching impact on the Big Ten,
Michigan and college football beyond this year's College Football Playoff race.

A look at the winners and losers from the scandal to date:

Michigan scandal winners and losers

Winners

Tony Petitti

Petitti certainly did not hit Michigan with a "slap on the wrist."
</item 10>

<item 11>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan noted that regarding evidence, there is nothing it is aware of to claim athletics personnel at Michigan committed the allegations, and said the rule
pertaining to electronic recordings of games is limited to the use of field equipment used by the institution for the game it is currently playing. It also said
Michigan has not seen much evidence, but rather has just heard a summary of second- and third-hand information.

The letter also said given the use of sign decoding by other universities, there is little reason to believe Stalions' efforts proved meaningful, and pointed
out the teams' average margin of victory since Stalions was suspended. It said the information obtained is "at most a reflection of information that teams can
and easily do obtain through public and permissible sources," and because of that it is "highly dubious" that Stalions' observations would have a "material
effect on the integrity of the competition."
</item 11>

<item 12>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
He has denied knowledge of the alleged scouting.

GO DEEPER Ten thoughts on the Big Ten's 2024 schedule

Required reading

(Photo: David Berding / Getty Images)
</item 12>

<item 13>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten said it would monitor the NCAA’s investigation into Michigan.

Advertisement

“The Big Ten is so much more powerful than the NCAA,” that third Big Ten coach said. “Why are you just sitting back and doing nothing about this? The Big Ten
can’t pound its chest for the last 30 years about how it does the right thing ethically (when other conferences like the SEC won’t) and then have this go on. If
this were a team in the bottom half of the Big Ten, would this be handled in the same way?

“When a running back gets hurt against Michigan because they knew exactly what play was coming, will that kid and his family have the ability to sue the Big
Ten?”

The NCAA investigation is ongoing, a process that typically moves slowly, which makes it difficult to imagine it will reach a resolution by the time the
postseason begins.
</item 13>

<item 14>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
To know exactly what the other team is doing, Michigan might as well have been playing with 15 guys on the field,” he said. “What’s the message the Big Ten is
sending now by doing nothing? Win now, pay later? We might as well just send people to (scout) their practices and their games. It doesn’t encourage anybody to
follow the rules. It’s just telling them to do the opposite and say, f— it.”

GO DEEPER Michigan should be punished, say 94% of CFB coaches in our poll. What else did it reveal?

The NCAA is investigating Michigan’s football program amid allegations that the Wolverines used illegal in-person scouting and the recording of signals to steal
signs this season. Ahead of the Michigan-Michigan State game on Oct. 21, the Big Ten approached MSU and said it was made aware of “credible evidence” regarding
the sign-stealing allegations. The Big Ten said it would monitor the NCAA’s investigation into Michigan.
</item 14>

<item 15>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 15>

<item 16>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Within it, the school warned Big Ten commissioner Tony Pettiti against levying "premature" disciplinary action against Jim Harbaugh and alleged that in-person
scouting and sign-stealing are commonplace across the NCAA.

"The conference should act cautiously when setting precedent given the reality that in-person scouting, collusion among opponents, and other questionable
practices may well be far more prevalent than believed," the letter read.

Michigan also included with the letter "two attached documents of Michigan's offensive and defensive signs allegedly stolen and shared by other Big Ten teams,"
per Yahoo Sports.

Conference awaits Michigan response

The Big Ten expects a formal response from Michigan to the notice of discipline by the end of the day Wednesday, ESPN's Pete Thamel reports. His colleague Adam
Rittenberg adds that a ruling is expected to come down Thursday.

Sources: The Big Ten expects a response from Michigan on the notice of discipline by the end of day on Wednesday. Don’t expect any ruling by the Big Ten on
Wednesday, as they are expected to take time to absorb Michigan’s response.
</item 16>

<item 17>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
… No doubt this all has had a profound effect.

“This guy’s being investigated for three different things now between the (alleged) illegal signal stealing, the (alleged) illegal recruiting during COVID and
that investigation into the offensive coordinator and alleged computer hacking. There are guys (on that call) who could lose jobs, and then there’s this guy
over here (Harbaugh) who is gonna get a new, bigger contract now, and they won’t do anything about him.”

Asked to describe the tone of the coaches’ sentiment expressed to Petitti, another Big Ten coach called it “angry” — particularly at the Big Ten’s lack of
action, or even apparent interest in taking any.

“Everybody’s upset,” that coach told The Athletic. “Why is nothing being done? We want to know, what else do you need to know to take action? We (the Big Ten
head coaches) want something done now.
</item 17>

<item 18>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 18>

<item 19>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
“They’re reading the tea leaves and wondering why the Big Ten hasn’t done anything yet. Every week and every day that goes by, people are like, ‘Something’s
gotta give.’ It’s getting a little bit out of hand when you see him (allegedly) on the Central Michigan sideline. The playing field is not level right now. How
can you have a team that you know has a competitive advantage over you still being allowed to play? That’s what the coaches are grappling with.”

Advertisement

“It feels like (former commissioner) Kevin (Warren) taking over and then COVID,” the source continued. “Tony’s walking into this situation, and people are
calling for the league to make a statement before they have all the facts.”

Despite frustration from all corners of the conference, sources at four different Big Ten schools said they do not expect the conference to levy any sort of
punishment against Michigan before the season ends.
</item 19>

<item 20>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
12

U-M president praises Wolverines for handling 'challenges and adversity'

The day after the Penn State win, University of Michigan President Santa Ono posted on social media, praising the team's ability to "respond to any challenge
head on with a conviction to do better and to emerge even stronger."

Countless members of the University of Michigan family have reached out to me over the weekend and I wanted to express my appreciation. Like any community, we
face our share of challenges and adversity. There have been many such moments in our history. But as our team showed so… — Santa Ono (@SantaJOno) November 12,
2023

The post drew immediate and extensive blowback on social media and beyond, as observers questioned Ono's tone with Michigan in the midst of a self-inflicted
controversy.

Saturday, Nov. 11

Warde Manuel issues strong statement against Big Ten

Michigan athletic director Warde Manuel did not mince words in a statement on the Big Ten's suspension of Harbaugh.
</item 20>

<item 21>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
The Big Ten does have the ability to act under its sportsmanship policy, but that doesn’t mean that it would want to act quickly or decisively before the NCAA
completes its entire investigation and allows Michigan a chance to respond to its findings.

This is an unprecedented situation; whatever Petitti decides to do (or not do) will set a precedent. The Big Ten itself doesn’t have investigators, so it needs
to rely upon the NCAA to do that part — and to determine who else was involved in the alleged scouting scheme. It’s not clear exactly what the coaches would
want the league to do to punish Michigan; banning the team from competing in the Big Ten championship, for example, would harm players who had nothing to do
with the sign-stealing apparatus.

A source briefed on the coaches’ call said Big Ten coaches are concerned about whether Michigan “should represent the Big Ten.”

“No matter what happens, if Michigan continues to move forward, the clouds will follow,” the source said.
</item 21>

<item 22>
Michigan’s Rivals Fume About Sign-Stealing: ‘This Was Damn Near Espionage’ (The New York Times, 2023-11-03T22:07:31+00:00)
Stalions bought tickets to games in at least seven Big Ten stadiums before those teams played the Wolverines over the past three seasons, including the 2023
season, sources told The Athletic last month. Purchasing the tickets is not a violation of NCAA rules, but using them to scout and record other teams would
violate the rules, prohibiting in-person, on-campus scouting and the audio or video recording of signals.

“They aren’t allegations. It happened,” Purdue coach Ryan Walters said Thursday night on his radio show ahead of Saturday’s game against the Wolverines.
“There’s video evidence. There’s ticket purchases and sales you can track back. We know for a fact they were at a number of our games. We’ve had to teach our
guys a new language.”

On Monday, coach Jim Harbaugh met with reporters and said “the people that know us the most think the most of us” as Michigan faces the NCAA investigation. He
has denied knowledge of the alleged scouting.
</item 22>

<item 23>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.
</item 23>

<item 24>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
“We went through and made sure we asked all the questions and got our compliance people involved. None of that came back at all. I can answer very strongly that
that did not happen.”

Michigan submits response to Big Ten

In the latest step toward a resolution, Michigan has submitted its response to the Big Ten's notice of discipline. ESPN reported earlier Wednesday that no
ruling from the conference is expected until Thursday, as the Big Ten is planning to "take time to absorb" what Michigan has to say.

Michigan has sent its response to the Big Ten, a school official tells ESPN. Commissioner Tony Petitti now must decide how to proceed with potential discipline.
— Adam Rittenberg (@ESPNRittenberg) November 8, 2023

Yahoo Sports obtained a copy of the 10-page letter sent by Michigan to the Big Ten.
</item 24>

<item 25>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.
</item 25>

<item 26>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 9, 2023

Big Ten commissioner to miss College Football Playoff meeting Thursday

Tony Pettiti is set to be absent from the College Football Playoff meeting set to take place in Dallas on Thursday. That leads to further speculation that
punishment could be coming soon in Michigan's sign-stealing scandal.

Big Ten commish Tony Petitti will NOT be in Dallas for today’s @CFBPlayoff meeting. — Heather Dinich (@CFBHeather) November 9, 2023

Wednesday, Nov. 8

Ohio State denies having shared Michigan's play signals with Purdue

Ohio State coach Ryan Day was asked about the allegations that Ohio State had given information about Michigan's signs to Purdue before the Big Ten championship
game in 2022. He denied them forthright.

“I can tell you this right now that nobody here did any of that,” Day said, per the Columbus Dispatch.
</item 26>

<item 27>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
He said while the program has been disappointed to hear about the allegations, it also feels that the conference skipped due process of a complete investigation
in suspending Harbaugh on Friday.

Manuel described the suspension as "completely unethical" and "an assault on the rights of everyone (especially in the Big Ten) to be judged by a fair and
complete investigation."

"All of the Head Coaches in the Big Ten (some who have been accused of actively participating in the trading of signals of opponents) and my Big Ten AD
colleagues can rejoice today that someone was 'held accountable,' but they should be worried about the new standard of judgment (without complete investigation)
that has been unleashed in this conference," Manuel said.

Statement from Michigan AD Warde Manuel. “I find that completely unethical, insulting to a well-established process within the NCAA, and an assault on the
rights of everyone (especially in the Big Ten) to be judged by a fair and complete investigation.”
</item 27>

<item 28>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 28>

<item 29>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
That ruling showed Petitti is not afraid to take on one of the Big Two within the conference. In a letter to Michigan, Petitti tried to dispel those notions.
Via The Athletic's Nicole Auerbach:

“The Conference does not play favorites among its members, nor does it take actions towards its members based on prejudgment or bias. Failing to act under the
extraordinary circumstances here could lead other Conference members to reasonably conclude that the conference has chosen to favor the University over all
other members.”

Tony Petitti, in letter to Michigan, says that the school initially refused to consent to the Big Ten obtaining the spreadsheet and other documents from the
NCAA related to its investigation.



Also makes point to say he's not playing favorites. pic.twitter.com/OgTvTNY1QI — Nicole Auerbach (@NicoleAuerbach) November 10, 2023

Most speculation pinned a potential Harbaugh suspension at two games knowing that the regular-season finale against No.
</item 29>

<item 30>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
17

Michigan fires coach connected to destroying evidence in investigation

Yahoo Sports' Ross Dellenger and Dan Wetzel reported on Friday that the NCAA's investigation into the sign-stealing and advanced scouting operation has found
that Stalions was at least partially funded by a Michigan booster, named as "Uncle T," and that linebackers coach Chris Partridge, who was fired by Michigan on
Friday, is alleged to have destroyed evidence on a computer pertaining to the sign-stealing allegations.

The report states Partridge is not believed to have had prior knowledge of the sign-stealing operation, but then later worked to cover it up. In a statement on
his firing, Michigan appeared to indicate a connection between Partridge and the NCAA's investigation, which Dellenger and Wetzel reported is still in the early
stage:

From the outset, our focus has been on seeking due process and allowing the NCAA to conduct a fair and deliberate investigation.
</item 30>
=== END ITEM id=6e41e7ab645cdc2f ===

=== ITEM id=afa0bc9dafb20db5 ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement also meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement also meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: The party described uses its own store.
  Fact 2.1: The party described uses its own payments.
  Fact 2.2: The party described forces developers to all have the same terms.
  Fact 2.3: The party described forces OEMs to all have the same terms.
  Fact 2.4: The party described forces carriers to all have the same terms.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 1>

<item 2>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
It’s not like how the watch uses its EKG capabilities to detect atrial fibrillation. On consumer wearables, SpO2 features generally have no medical use cases.
They’re essentially for “entertainment” purposes, and we’ve repeatedly reported that you should never, ever use them as any sort of replacement for traditional
pulse oximeters. The only real blow for Apple is if the rumors are correct and it’s working on sleep apnea detection. Losing the sensors in that instance would
make developing this feature harder. However, that feature would require a lengthy FDA clearance process. That means more advanced health features like sleep
apnea detection are more of a medium to long-term concern — and it’s very possible that Apple will have a workaround by the time such features are ready for
consumers.

That said, for this route to be successful, Iancu says it depends on the specific patent, ITC ruling, and what exactly Apple has planned as far as redesigning
the product.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 3>

<item 4>
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
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 5>

<item 6>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
Listen to TC+ editor Alex Wilhelm and I talk more about how there’s plenty of market share to go around in Friday’s episode of the Equity podcast below. — Mary
Ann

PayPal is being sued for alleged “Draconian” payment policies

On October 5, Mary Ann broke the news that PayPal has been hit with a class action lawsuit by consumers represented by law firm Hagens Berman alleging that the
fintech giant’s anti-steering rules stifle competition against lower-cost payment platforms such as Stripe and Shopify. Specifically, according to an
investigation conducted by the firm’s consumer rights attorneys, PayPal has subjected consumers to excess charges when purchasing from online merchants that
accept PayPal or Venmo.

Someone commented on social media that they didn’t see the problem with what PayPal is doing and questioned whether or not Visa and Mastercard do the same
thing. So I went back to the attorneys who filed the lawsuit, who noted that the “anti-steering rules” are not the first of their kind.
</item 6>

<item 7>
The Apple Watch’s double tap gesture points at a new way to use wearables (The Verge, 2023-10-25T17:04:01+00:00)
That lack of choice is also apparent in third-party apps. My email app lets me double-tap to start a quick reply using Siri, but even in 2023, I am not
dictating my emails. Apps like Spotify and Pocket Casts don’t work with double tap, either. Most third-party apps that do work with double tap just let you
dismiss notifications. That’s helpful but still limiting.

With watchOS 10, apps like Fitness and Weather have been redesigned to be more glanceable. Instead of one long screen that you scroll endlessly through,
information is divided into more digestible chunks. You still scroll, but there’s less of it. This is great, except I often want to use double tap to navigate
through menus instead of the digital crown. And I can’t.

You can scroll through the new widget Smart Stack, but if I want to select a widget and open it, I still have to use my other hand.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 9>

<item 10>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
But in this case, the important thing is that the ITC sided with Masimo. In January 2023, it ruled that Apple Watches did in fact infringe on Masimo patents.

The ITC then issued an import ban in October, as well as an order to stop selling products infringing on Masimo’s patents that had already been imported. We’re
currently nearing the end of a 60-day presidential review period, in which President Joe Biden or the US Trade Representative (USTR) has the opportunity to veto
the ban. If there’s no veto by the time the review period ends, the ban will go into effect.

But will Biden actually veto?

Apple getting a presidential veto would be like lightning striking the same place twice.
</item 10>

<item 11>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
They said: “Visa and MasterCard once imposed similar anti-steering rules on merchants accepting their cards but, after the Justice Department sued the networks
for antitrust violations, they agreed in 2010 to eliminate their anti-steering rules as part of the settlement. With payments transitioning into the digital
realm, PayPal has simply ripped a page right from the Visa and MasterCard [sic] playbook.”

Meanwhile, Patrick McGahan, a partner at Scott+Scott who focuses on antitrust litigation, had an interesting take that actually involves both of those card
giants. He told TechCrunch that the case illustrates that “the tensions between merchants and the payment system providers are not over, and that litigation
regarding this key cost faced by merchants is likely to continue for some time.”

He added: “Platform companies that operate as gatekeepers in their market, such as PayPal, will continue to be the subject of antitrust litigation as a result
of the significant fees they charge.
</item 11>

<item 12>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 12>

<item 13>
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
</item 13>

<item 14>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 14>

<item 15>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 15>

<item 16>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
“From a legal perspective, whatever change Apple makes has to be significant enough to avoid the patent, and it cannot be what’s called a ‘colorable’
difference. Minor changes will not be sufficient,” Iancu says.

Basically, Apple can’t simply make a small tweak to the code and call it a day. Masimo claims that the patents are hardware related, so even completely
disabling the sensor may not work depending on how the patent is written. Apple could remove it entirely from the design, but that would require mucking around
with Apple’s meticulous supply chain and could take some time.

Then there are the options that would cost Apple some money. Apple could sit down with Masimo and hammer out a licensing deal for the infringed patents. In a
New York Times interview, Masimo CEO Joe Kiani said Apple “had not engaged in licensing negotiations.” It’s an easy-seeming fix, so why not do it? For starters,
both parties have to want to settle.
</item 16>

<item 17>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Where does Apple go from here?

A hail mary veto is unlikely, but that doesn’t mean Apple is just going to accept an import ban on a $17 billion segment of its business. Apple spokesperson
Nikki Rothberg told The Verge in a statement that the company was “pursuing a range of legal and technical options to ensure that Apple Watch is available to
customers.” That means the watch is going back on sale one way or another — it’s just a question of what path Apple takes.

Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months. The company will also probably ask for a
stay on the import ban, but Brittingham says that’s not likely to be granted because Apple would have to prove that it would be irreparably harmed. In other
words, it’d have to make a convincing case that it would go out of business if it can’t sell the Apple Watch.

Apple says it’s exploring both legal and technical options to get around this import ban.
</item 17>

<item 18>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.
</item 18>

<item 19>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Like Masimo, AliveCor argued that Apple stole its tech to develop the wearable EKG feature, thereby infringing on its patents. However, while the two cases look
similar, the result of one doesn’t speak to how the other will play out.

“Every case is different, every patent is different,” says Levi. “Different complainants who bring up a case have got their own businesses, their own products,
and those implicate a whole different set of facts.”

The Apple Watch SE will remain unaffected because it doesn’t have a blood oxygen sensor. Photo by Amelia Holowaty Krales / The Verge

Case in point, while the ITC granted AliveCor’s import ban, it hasn’t had any impact because the Patent Trial and Appeal Board ruled that AliveCor’s EKG tech
isn’t actually patentable. AliveCor would have to win its appeal to that ruling before any Apple Watches with EKG features would be pulled from the market for
that reason.
</item 19>

<item 20>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 20>

<item 21>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 22>

<item 23>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
For starters, both parties have to want to settle.

“Apple is a tough nut to crack,” says Brittingham, noting that Apple is one of the most sued companies in Silicon Valley. The company’s large cash reserves also
make taking Apple to court a costly endeavor. Masimo, for example, has reportedly spent $60 million litigating the issue thus far. “I think they tend to hold
out as long as possible in order to make it less appetizing to sue Apple.”

The other option is for Apple to avoid the import ban altogether. Iancu says Apple could simply decide to manufacture the watches within the US instead of
overseas — but that’s a significant supply chain change that’s unlikely to happen any time soon.

Does this set a precedent?

This import ban isn’t the only one Apple is facing. In February, the Biden administration also declined to veto another ITC import ban and cease-and-desist
order stemming from a dispute with AliveCor over the Apple Watch’s EKG features.
</item 23>

<item 24>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Soon after, Apple reportedly poached several Masimo engineers and its chief medical officer. And then in fall 2020, Apple released the Apple Watch Series 6 —
its first Apple Watch to feature an SpO2 sensor to measure blood oxygen saturation levels.

The SpO2 sensors measure blood oxygen levels and are at the heart of this patent dispute. Photo by Amelia Holowaty Krales / The Verge

In 2020, Masimo filed a lawsuit in the US District Court in the Central District of California accusing Apple of stealing trade secrets and infringing on 10 of
its patents. That case dragged on and on, so Masimo then filed a separate case with the ITC in 2021. Then last year, Apple sued Masimo back, claiming the
company made an Apple Watch clone with its Masimo W1 Medical Watch. Clearly, there’s some bad blood here. But in this case, the important thing is that the ITC
sided with Masimo.
</item 24>

<item 25>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 25>

<item 26>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Apple getting a presidential veto would be like lightning striking the same place twice.

“It is extraordinarily rare for the President to overturn a decision from the International Trade Commission and in my view, [it’s] unlikely in this particular
case,” says Andrei Iancu, co-chair of the Council for Innovation Promotion (C4IP) and partner at Sullivan & Cromwell LLP, who was also the director of the US
Patent and Trademark Office from 2018 to 2021.

To get a veto, a company usually has to show a public interest or health policy basis — and that isn’t the case here. “It’s unlikely that the import exclusion
order will be disapproved by the USTR and the president,” says Smith Brittingham, partner at Finnegan, Henderson, Farabow, Garrett & Dunner, LLP. “The only real
answer would be, ‘Well, it’s Apple.’”

That said, Apple did manage to beat the system once. In 2013, then President Barack Obama vetoed an import ban on the iPhone.
</item 26>

<item 27>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 27>

<item 28>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
Earlier this year, the US International Trade Commission (ITC) ruled that Apple had infringed on two patents from medical device maker Masimo. As a result, the
ITC said it would impose an import ban on the Apple Watch Series 9 and Ultra 2 starting December 26th. At the time it was hard to believe that this would
actually come to pass: Apple gets sued all the time, and even when it loses, how often does it actually face dramatic consequences?

Well, now would be one of those times. Experts say that, barring a Christmas miracle, it’s unlikely that Apple will find a way to escape the ban. Case in point,
the company shocked everyone yesterday when it decided to preemptively pull the watches from its online store starting December 21st at 3PM ET. And after the
24th, they’ll disappear from Apple Stores, too.
</item 28>

<item 29>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 29>

<item 30>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?
</item 30>
=== END ITEM id=afa0bc9dafb20db5 ===
