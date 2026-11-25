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

=== ITEM id=aa5c6e7ab6d2494b ===
Gold sentences and facts:
Sentence 0: After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry at his family’s response to it.
  Fact 0.0: William learned of Diana’s death from his father.
  Fact 0.1: William learned of Diana’s death at the royal residence of Balmoral in Scotland.
  Fact 0.2: The character William is played by Rufus Kampa.
  Fact 0.3: William is left heartbroken with grief.
  Fact 0.4: William is angry at his family’s response to Diana’s death.
Sentence 1: The latest installment takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
  Fact 1.0: The latest installment takes place between the late 1990s and the mid-2000s.
  Fact 1.1: The latest installment includes the lead-up to Princess Diana’s fatal Paris car crash.
  Fact 1.2: Princess Diana’s fatal Paris car crash happened in 1997.

Retrieved items:
<item 1>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
“And if your husband is close enough to your daughter, he could just ask her himself. It probably isn’t as personal for her as it is for him, especially if you
and your husband did not have any more children.”

“This isn’t about him and all he’s done. It’s about Calla,” another added. “She lost her dad and, while the grief fades a bit over the years, big events bring
it roaring back in (at least that’s been my experience). Her wedding is going to be bittersweet. She needs her parent (the one she’s always had and the one who
loved her dad) to ‘give her away.’”

A third wrote: “The real question here is why does your husband believe he’s more entitled to this honour of walking your daughter down the aisle, and ‘giving’
her away than YOU? You, who is her mother, who has been there through it all, who has comforted and given her a home and place to be herself, unconditionally?”
</item 1>

<item 2>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
"We did not choose to release these prisoners who have blood on their hands. That's a disgusting accusation."

The strange question aroused anger and criticism worldwide but Kate Burley did not apologize and even justified her question on X (formerly Twitter) by saying
that interviewers sometimes present one side of an argument so that those being interviewed can refute it. She claimed that she was presenting a controversial
opinion that she had heard to allow the interviewee to respond.

Levy does not accept this explanation. "It's true that the role of the media is to challenge speakers with controversial opinions but it is also to filter out
the bullshit and to know when you have been told nonsense that requires no further comment," he says.

Levy posted the clip from the interview on his Twitter account and it received 16 million views and 15,000 shares (just from the Tweet). His number of followers
has soared from 24,000 before the war to 140,000 after the Sky News interview.
</item 2>

<item 3>
‘I see myself as a royal’: artist Vincent Namatjira on colonialism, satire and his great-grandfather’s legacy (The Guardian, 2023-10-16T14:00:44+00:00)
For more information see our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after
newsletter promotion

“To be honest, I would like the government to apologise [for] taking these young Indigenous people into foster care,” he says. “In reality, we are losing much
more of where we’re originally from, and our background, and our history. We have lost a lot of things, our culture, languages and our tradition.”

Indigenous people, we want to be heard and to take a stand Vincent Namatjira

Among prime ministers, royals and chief executives, painted portraits typically convey the power of their position – officially commissioned for palaces,
parliaments or boardrooms, with artists granted rare access to their famous sitters. Vincent’s portraits, on the other hand, painted in his studio at Iwantja
Arts in Indulkana, are gleefully unofficial and defiant.
</item 3>

<item 4>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
When Ser Alliser Thorne drove a dagger into the heart of Game of Thrones’ Byronic hero Jon Snow in the final episode of Season 5, he left the show’s 10 million
viewers in a profound state of shock. “It’s true, I have started to play a lot of dark characters,” he says, smiling. “And at first it was a great challenge
because I’m quite a sociable fellow.” So, how does he find that darkness? “It’s a sense of nothingness. In all honesty, I’m not very good at being completely on
my own,” he explains. “I come alive when I’m with other people. So staying in hotel rooms away from my family isn’t great and it doesn’t get any easier. There’s
something empty and a bit scary within me when I’m not with people. I think that’s where that stuff comes from.” Owen Teale unleashes the darkness as Ser
Alliser Thorne in Game of Thrones.
</item 4>

<item 5>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
From there, things move faster but the kids were on board for things to move on.”

The mother specified that it was clear to her children that her husband “was not going to be filling the role of a dad in their lives”. She added that her
partner “was fine with that, as long as he was respected and they could work toward being a caring family”.

She continued to note that while she had a close relationship with Calla, it was “extremely tough” for her child to lose her father. She specified that while
her child has a good relationship with her stepfather, “it comes nowhere close to the bond she had with her dad”. She also shared that she continued to get
close to her daughter after she turned 12, before recalling a point in Calla’s childhood when she asked her mother to walk her down the aisle.
</item 5>

<item 6>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
The Independent has contacted u/PopShort7064 for comment.
</item 6>

<item 7>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
However, during an appearance on the Today show on 16 October, Jada clarified to Hoda Kotb that she and the Suicide Squad actor are “working hard” and are
“concentrating on healing” to get their relationship to a better place.

“There’s no finding another great love, and I think that’s the point,” she said. “It’s like we are in a place now that we are in a deep, healing space. And we
are really concentrating on healing the relationship between us.”

The Girls Trip star continued: “There’s no divorce on paper. We really have been working hard. That’s the whole thing. We are working very hard at bringing our
relationship together. Back to a life partnership.”
</item 7>

<item 8>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
Save Log in , register or subscribe to save articles for later. Save articles for later Add articles to your saved list and come back to them any time. Got it
Normal text size Larger text size Very large text size It all started with a car full of booze and a bear costume. Owen Teale, recently suspended from his
sixth-form college for organising a riotous Christmas party for fellow students, was in need of a job. “They held an investigation into the party and I was the
first to be called in,” says the 62-year-old Welsh actor best known as gimlet-eyed master-at-arms Ser Alliser Thorne in Game of Thrones. “They’d found the empty
bottles and people were falling over drunk. I said, ‘yeah, I did it. It was me. I borrowed my mother’s car, loaded it up with booze and smuggled it in’.” Teale
had no intention of joining his father at the limestone quarry that fed the steel works at Port Talbot.
</item 8>

<item 9>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

The first four episodes of The Crown season six are officially here, but many viewers believe one scene has depicted the most devastating moment throughout the
entire series.

On 16 November, Netflix dropped the first four episodes of The Crown’s final season, followed by the last six episodes on 14 December. The latest installment
takes place between the late 1990s to mid-2000s, and includes the lead-up to Princess Diana’s fatal Paris car crash in 1997.
</item 9>

<item 10>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
After months of court discussions, Federline rejected Spears’s child support proposal and instead doubled his request to $40k a month. A settlement was finally
reached between the two parties in September 2018, with Spears increasing her child support payments to Federline from $20,000 to $35,000 a month.

Amid her father’s health crisis and ongoing emotional distress, Spears voluntarily checked into a mental health treatment facility in 2019. After hearing the
news, Federline’s attorney Mark Kaplan told Entertainment Tonight: “Kevin commends Britney for recognising that she needs to take a step back and that she is
taking the reasonable steps to dealing with her situation in a responsible way. He wishes everyone well.” That same year, Federline was granted 70 percent
custodial rights and Spears 30 per cent.

After years with her father in control of her finances and personal life, Spears was finally granted freedom from her conservatorship in November 2021.
</item 10>

<item 11>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
Teale has the grace to admit he enjoys this new level of celebrity. “People seem to realise that without the darkness of people like Ser Alliser, Jon Snow
couldn’t have shone as brightly as he did.” Owen Teale with Clare Homan in The Fifteen Streets. Credit: ABC He was well into his 50s when he landed his highest
profile role, but it would be wrong to think of Teale as a late starter. In 1989, not long out of drama school, he starred in the hit television adaptation of
Catherine Cookson’s romance The Fifteen Streets. The cast included Sean Bean, another young working-class actor, who went on to become a leading man in
Hollywood before playing Ned Stark in Game of Thrones. I ask Teale if he ever wishes he’d followed Bean’s trajectory. “Oh god, yeah,” he replies. “Absolutely.
You instinctively think [playing leading men] will lead to feelings of success. But it never came my way.
</item 11>

<item 12>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
"With drugs, he was a monster."

Williams began to weave cocaine into his stand-up jokes — which many laughed at, despite the light it shed on how deep the comedian was into his cocaine use.

"I can't tell you how many times Robin came onstage at The Comedy Store and a big part of his routine was cocaine and the drugs that he had done," Howie Mandel
recalled. "And even though everybody was laughing, and it was really funny, he was really doing a lot of cocaine. He found the funny side of it, but ultimately
it was real and really, it's not funny, but that's who he is."

LIKE WHAT YOU’RE READING? CLICK HERE FOR MORE ENTERTAINMENT NEWS

John Belushi's death in 1982 "threw cold water" on Williams, who had been close to the comedian. Belushi died from a drug overdose in a bungalow at the Chateau
Marmont hotel in West Hollywood, California.
</item 12>

<item 13>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Do the explanations produce greater acceptance?

"Yes. There is understanding that we must end this war by destroying the capabilities of Hamas, otherwise we will be in the same situation in another six
months. This is something I explain a lot to the Irish, for example. The Irish have been pressing for a ceasefire for a long time and I explain to them that if
we had taken their advice and stopped firing then little Emily Hand would still be in captivity, and if we take their advice now to agree to a ceasefire then
Hamas will commit more atrocities, because it will feel empowered and it will come out of this war with a feeling that the world has its back.

"We are being asked difficult questions about the dead on the other side and we must remind them that everyone who was killed in the Gaza Strip over the last
month would be alive today if Hamas had not started the atrocities on October 7, and then decided to fight in populated areas, while preventing people from
leaving despite all of Israel's efforts for over a month to evacuate civilians from the battle zones."
</item 13>

<item 14>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
She claimed that three weeks after the incident, her husband hasn’t changed his stance and he still feels like his wife “was wrong”. She also said that he told
her he “felt he deserved more from all” of the family, and that his wife should be “making sure he is respected and honoured for his contribution to the kids’
lives”.

The Reddit post has quickly gone viral, as it has received more than 6,500 upvotes. In the comments, many people have gone on to defend the bride’s mother,
noting that her husband should respect Calla’s decisions for her wedding day. They also criticised the stepfather and claimed he should be more considerate of
Calla’s bond with her late father.

“I can tell you that no man, as you know, could ever hold a candle to my father so I feel comfortable saying that your daughter would probably never ask him
[because] it would be like disrespecting the memory of her father,” one wrote.
</item 14>

<item 15>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
RIP Bill."

Andy Burnham, Mayor of Greater Manchester and Everton fan: "It does feel like the end of an era, Bill was a big-hearted person. Such generosity and backed so
many people that needed help, perhaps people did not see that in the media.

"He had a deep love for his club and the city he was from, it is an emotional night."

Actor Sir Ian McKellen: "Like many grateful actors I am in debt to Bill Kenwright for employment. He seemed to have known everyone in the business and to care
about them. Yet every chat would veer round to his equal passion - Everton."
</item 15>

<item 16>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“Whether he realises it or not, Mr Federline has not only violated the privacy and dignity of the mother of his children, he has undermined his own children,
whose privacy he should protect,” the statement read. “Putting aside his ITV interview, Mr Federline’s ill-advised decision to post an old video of his 11 and
12-year-old children was cruel, bottom of the barrel stuff. It was abhorrent.”

“In addition to demeaning himself and violating societal norms, he has now also created various legal issues for himself, including, but not limited to,
implicating cyber-harassment and cyber-bullying statutes, among other things.”

The Woman in Me hits shelves in the US and UK on 24 October.
</item 16>

<item 17>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
With stories in The Age and Sydney Morning Herald, as well as on 60 Minutes, journalists Nick McKenzie and Chris Masters pushed to bring the truth to the
public, navigating divisions in the SAS and eventually the defamation trial that Roberts-Smith brought to court and lost. It’s been a compelling saga, and Dora
Weekley’s documentary should include hitherto unknown details and unheard commentary.

Advertisement

I still haven’t got over the nightmarish sounds of a spinal surgery going very wrong in the first season, but nonetheless Dr Death (December 22) is back for a
second serve of malpractice and monstrous medical egos. This American true crime drama is now an anthology. Every season is a different doctor whose abuse of
privilege posed a criminal risk to their patients. The new season will cover Paolo Macchiarini, an Italian surgeon and researcher who was accused of laboratory
fraud and manipulative behaviour.
</item 17>

<item 18>
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
</item 18>

<item 19>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
Owen Teale unleashes the darkness as Ser Alliser Thorne in Game of Thrones. His ability to tap into the void served him well as he prepared to enact the murder
of Snow, played by Kit Harington. “There was Kit … who was so beautiful. And he’s such a lovely man. By connecting to this feeling of emptiness, of having
nothing inside of me, I was able to look at him and everything he had - his looks, his grace - and feel pure hatred. For all of Ser Alliser’s explanations [for
the murder], deep down he just has a hatred of Jon Snow’s popularity and gifts.” Like Iago, I suggest. “Yes,” he says. “Very like Iago.”

Advertisement

When Snow’s death was broadcast, Teale braced himself for the fall-out. He imagined being confronted by irate Game of Thrones fans who would hate him for
“killing the goodness”.
</item 19>

<item 20>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
However, she grew to become closer to Camilla. Despite announcing in 2005 that Camilla would become the Princess Consort rather than Queen Consort, it was
announced that Camilla would be queen seven months before Elizabeth’s death.

Access unlimited streaming of movies and TV shows with Amazon Prime Video Sign up now for a 30-day free trial Sign up

Access unlimited streaming of movies and TV shows with Amazon Prime Video Sign up now for a 30-day free trial Sign up

Dodi Fayed’s engagement to an American model

When viewers first meet Dodi in The Crown, he’s in Paris with a woman with short, blonde hair. However, it’s not Diana, but an American woman who he says he’s
picking out fabric for a Malibu home with and says he’s marrying in three weeks. His father Mohamed is unimpressed, calling him to Saint Tropez to meet a
“special guest” and leave his “gold-digger” fiance – who he also calls “Madame Bikini” – behind.
</item 20>

<item 21>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Recalling their discussion to the Queen, Blair says that Diana feels she “still has a lot to offer the country as a public servant”. He says that she had asked
if there was a way to work with the government on “a more formal basis” and that any “official role” would be appreciated.

Bertie Carvel as Blair (Netflix)

Following Diana’s death, Downing Street confirmed that Diana had visited Blair at Chequers with her son a few weeks before, where they discussed a special role
for her as an overseas “ambassador” for Britain. Diana told a journalist at the time that the prime minister had recognised her skills and asked her to
undertake “missions” abroad for Britain.

Queen Elizabeth’s reluctance to accept Camilla

As The Crown season six begins, Charles is frustrated at the public for demonising Camilla long after his split from Diana.
</item 21>

<item 22>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
“She was 15 when she told me if she ever got married she would want me to walk her down the aisle and for us to do a mother/daughter dance instead of a
father/daughter one,” the parent wrote. “I told her we could dance to the song her dad used to sing to her. She said she loved the idea.”

The Reddit user expressed that she “always assumed things would change” down the line, with her daughter either choosing to walk alone or with her brother, “as
a representation of her dad”. However, she shared that when her daughter got engaged, she “asked [her mother] right away” to walk her down the aisle.

“She brought up our past conversation and told me she wanted exactly that. I told her I would be honoured,” she wrote. “We cried tears of joy together. I told
her that her dad would be so proud.”

However, according to the Reddit user, her husband didn’t take the news as expected.
</item 22>

<item 23>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
The world of “Suburra” takes a new turn.

Available Nov. 15

Feedback (Netflix Series)

From Netflix: A former rock star with a faulty memory and a family in shambles, all due to his addiction problem, sets off on a frantic search to find his
missing son.

First Wives Club: Seasons 1-3

Matt Rife: Natural Selection (Netflix Comedy)

From Netflix: A new stand-up special from comedian Matt Rife.

Million Dollar Decorators: Season 1

Available Nov. 16

Best. Christmas. Ever! (Netflix Film)

From Netflix: After a twist of fate brings their families together for Christmas, Charlotte sets out to prove her old friend Jackie’s life is too good to be
true.

The Crown: Season 6 Part 1 (Netflix Series)

From Netflix: Queen Elizabeth II reflects on her legacy and lineage as Diana dazzles the public in the final weeks of her life and the monarchy faces a
reckoning.
</item 23>

<item 24>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
He went on to explain that he’s “happier than [he’s] ever been in [his] entire life” and praised Jada for continuing to be by his side.

“I have achieved every single thing I have ever wanted in my entire life. There is not one thing left that I desire in this world - all of the awards, all of
the money, the family - everything I’ve ever dreamed,” he said. “And those dreams were largely built on the foundation of Jada’s sacrifices. There were many
times when Jada put her career aside so I could follow the dreams of mine.”

The Fresh Prince of Bel-Air star thanked his wife for putting him “before herself” and for maintaining a good relationship with his ex-wife Sheree Zampino. Will
and Zampino share a 30-year-old son, Trey Smith, who was also present for the book event at the Enoch Pratt Free Library.
</item 24>

<item 25>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Per a Deadline report, NBC accumulated an average of 27m viewers, with a total of 29m tuning in during the second quarter.

“That’s gotta be an NFL record right there,” Jason told his younger brother, as he noted that about two million viewers were estimated to be female. “I think a
few of them might have been there for Deadpool and, uh, Taylor.”

In the YouTube comments under the New Heights video podcast, fellow fans shared their opinions on the NFL’s coverage. “I agree the NFL needs to calm down.
Pissing off football fans is not how you repay your boi [sic] for all he’s done for the sport,” one person wrote.

Another remarked: “Pity Taylor. All she ever wanted was to watch the football game. Now she’s being blamed and hated left and right.”

“The NFL needs to seriously tone it down and let Trav and Tay explore this beautiful thing they have going on in peace,” someone else said.
</item 25>

<item 26>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 21.39 GMT The difference two days make Tuesday, Tarouba England 267-3, 19 sixes

Thursday, Taroura England 132 all out, five sixes

Updated at 21.45 GMT

6d ago 21.36 GMT WICKET! England 132 all out (Curran c Rutherford b Holder 12) Sam Curran clouts the next ball to long-off to end an increasingly miserable
England innings. They were 109 for four after 14 overs, and then the house fell down. West Indies need 133 to win the series.

Updated at 21.38 GMT

6d ago 21.33 GMT WICKET! England 132-9 (Rashid run out 1) Adil Rashid sacrifices himself, just like in the 2019 World Cup final, so that Curran can stay on
strike. Four balls remaining.
</item 26>

<item 27>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
The singer made a powerful testimony at her conservatorship hearing that June, where she spoke out for the first time about her experience, calling for those
involved to be “put in jail”. Following the testimony, Federline expressed his support for the singer, saying he “only hopes the best for her because when the
best for her is achieved, it’s the best for their kids.”

“The best thing would be for their mom to be healthy and happy,” his lawyer said in a statement at the time. “And if either of those things aren’t true, it
doesn’t provide for the best setting for custody to be exercised.”

Britney Spears poses with her sons Jayden James and Preston at Dodger Stadium in 2013 (Getty Images)

In June 2022, Spears married her longtime boyfriend Sam Asghari during an intimate ceremony at their Los Angeles home, just one month after the singer revealed
she suffered a miscarriage.
</item 27>

<item 28>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Back in October few people had heard of Levy. But now sitting in a café in the Azrieli Mall in Tel Aviv to be interviewed by "Globes," passersby clearly
recognize him and some even stop to take a selfie with him.

To filter out the bullshit

In recent weeks, interviews with Levy have gained massive exposure worldwide and the Israeli public has embraced the eloquent and quick-witted spokesperson who
has been articulating Israel's point of view so admirably, with his English accent. With degrees from both Oxford and Cambridge, Levy received the greatest
exposure in an interview with UK Sky News presenter Kay Burley, who asked him whether the fact that Israel is willing to release 150 Palestinian prisoners in
exchange for 50 Israeli hostages indicates that Israel thinks that Palestinian lives are worth less.

Levy's spontaneous, shocked response saw his eyes open wide with surprise and his eyebrows shoot up. But within a second he had regained his composure and
explained that Israel would release one Palestinian for each Jewish hostage if it could.
</item 28>

<item 29>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It’s just a bit mad here. I don’t really understand how I ended up here. Mummy just needs to make some changes to her life, that’s all. But that’s not your
problem, that’s mine.”

Before saying their goodbyes, Diana reminds William and Harry that she will be seeing them the next day - although that day never comes. “I’ll see you both
tomorrow night,” she says. “Just the three of us.”

The moment unsurprisingly tugged on the heartstrings of many viewers, who took to social media to share their thoughts on the devastating moment.

“The last phone call between Harry, William and Diana broke my heart,” one fan wrote on X, formerly known as Twitter.

“That precious phone call between Diana and her sons,” said someone else. “Don’t mind me, I’m just gonna cry in one corner.”

“The last call between Harry, William and Diana is heartbreaking,” a third user commented.
</item 29>

<item 30>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
It was later believed to have been delivered to their room before they went to Dodi’s flat, where a ring bearing the words “Dis-moi Oui” (“Tell me Yes”) – also
the name of episode three – was later recovered, alongside a receipt for a “bague de fiançaille” (engagement ring). Speaking at the inquest, his father Mohamed
claimed that the pair had met the jeweller in Monte Carlo during the holiday on the yacht, and that he believed Dodi planned to propose that night.

Prince William goes missing in Balmoral after Diana’s death

After learning of Diana’s death from his father at the royal residence of Balmoral in Scotland, William (Rufus Kampa) is left heartbroken with grief and angry
at his family’s response to it. At one point, Charles is informed that the prince is not in his room and that “no one can find him”, prompting Charles and
Prince Harry to look for him and a subsequent hunt across the grounds for the young royal.
</item 30>

<item 31>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
However, Williams continued to do drugs after his stand-up shows.

"When you did have that kind of fame, drugs were just given to you," Hecht noted.

ROBIN WILLIAMS' KIDS PAY TRIBUTE TO THE LATE ACTOR ON WHAT WOULD'VE BEEN HIS 72ND BIRTHDAY: ‘LOVE YOU FOREVER’

Filmmaker and comedian Mike Binder recalled Williams' reaction to "a gram of coke" while the two were out in Hollywood in the '80s.

"Robin said, ‘Let me take that. Do you mind if I grab a hit off that in the bathroom?’" Binder recalled in the documentary. "He came back, and it was empty. It
was like, ‘Whoa.’ It was like 8:15 at night."

"It was an accident, I'm sorry," Binder said the comedian responded after being called out for taking the entire gram.

"With drugs, he was a monster."
</item 31>

<item 32>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 32>

<item 33>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Chelsea were given a penalty and Tottenham went down to 10 men. #TOTCHE pic.twitter.com/sOzTrxgBzm — NBC Sports Soccer (@NBCSportsSoccer) November 6, 2023

Watch in UK:

"AMID FRENZY AND FURY!" 🤯



Cole Palmer levels for Chelsea as Cristian Romero sees red! 🟥 pic.twitter.com/1oULwk1uOP — Sky Sports Premier League (@SkySportsPL) November 6, 2023

Nicolas Jackson goal disallowed for offside

Chelsea would see a third goal of the half disallowed as Nicolas Jackson put the ball in the net on 37 minutes but Raheem Sterling was offside in the buildup
just past the midfield stripe. There's very little argument to be had here, and the game played on.

While it was a decision that featured little, if any, controversy, it added to the mayhem of the half and contributed to the game's overall narrative.
</item 33>

<item 34>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Sign up to our free IndyArts newsletter for all the latest entertainment news and reviews Sign up to our free IndyArts newsletter Please enter a valid email
address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our privacy notice
Thanks for signing up to the

IndyArts email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

As Netflix’s acclaimed royal drama moves closer to the present day, the line between fact and fiction feels ever more blurred.

While Peter Morgan’s show has always presented a dramatised take on real-life events, many have argued that viewers struggle to tell the difference between the
real royal family and the events on the show. When season five arrived last year, there were calls for the show to include a “fictional explainer” – something
Netflix did not do.
</item 34>

<item 35>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Speaking to the crowd, Will said that Jada’s positive relationship with Zampino allowed all three of his children the chance to grow up together.

“Can you show up and love somebody for the rest of your life, even when you don’t agree with them?” he asked. “Jada is the best friend I have ever had on this
planet and I am going to show up for her and support her for the rest of my life.”

This isn’t the first time that Will has opened up about his relationship, as it’s continued to make headlines. Earlier this week, he appeared to mock the
publicity storm surrounding him and Jada in a video titled “Official statement” shared on Instagram. “So, here’s the thing. My opinion is... uh…” the actor
said, before sneezing into the crook of his arm. As he sneezed, the video comically zoomed out to show several different landscapes including lakes and
mountains.
</item 35>

<item 36>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 36>

<item 37>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
“My wife was about to give birth, so I said ‘I’m off’. I’m sure part of it was I just didn’t believe that I was going to waltz into leading man roles in
Hollywood.” And there it is, that ambivalence. Despite all the plaudits, the awards and the steady offers of work, you sense Teale has never been certain of his
gifts. It’s probably why he’s so consistently good – complacency isn’t in his nature. Owen Teale with Toni Collette in Dream Horse: “Toni just raised my game,”
he says. Credit: AP He’s far better at recognising brilliance in others. For example, Toni Collette, who he worked with on Dream Horse, a 2020 movie about an
unlikely champion racehorse. “Toni just raised my game. That’s a huge talent, huge,” he says, shaking his head in wonder.
</item 37>

<item 38>
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
</item 38>

<item 39>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
His recent comments come after Jada made major revelations about their marriage in her memoir, including how she and Will have been separated for seven years.

During the event in Baltimore, Will began by speaking to the crowd about his love and admiration for his partner, who he married in 1997. “I just really wanted
to come out and just be here and hold it down for you the way you have held it down for me,” he said during the event, as reported by The Baltimore Sun.

The King Richard star went on to describe the challenges and achievements he’s had throughout his marriage, adding: “We have had a very, very long and
tumultuous [relationship]. We call it ‘brutiful.’ It was brutal and beautiful at the same time."

At another point during the event, Will referred to his union with Jada as a “sloppy public experiment in unconditional love,” according to the Baltimore
Banner.
</item 39>

<item 40>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.
</item 40>

<item 41>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
London derbies are always full of venom and vigor, but the opening 45 minutes of Tottenham vs. Chelsea on Monday produced one of the greatest train wrecks in
the history of the English top flight.

With repeated refereeing blunders one of the main talking points of the season thus far, a half full of VAR checks and highly scrutinized decisions caused
absolute chaos on the Tottenham Hotspur Stadium pitch.

All told, one red card and four yellow cards were produced by Michael Oliver with help from VAR official John Brooks, while Chelsea saw goals ruled out on an
incredible four separate occasions. It resulted in an incredible 12 minutes of stoppage time, and while many of the decisions were widely analysed as correct,
the lengthy review processes were criticised just as much as the eventual results.

The first half would end level at 1-1, with Tottenham reduced to 10 men, and Premier League fans had plenty to talk about with friends, fans, and coworkers.
</item 41>

<item 42>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
Instead, he was asked for endless selfies and impromptu recitations of Thorne catchphrases such as “For the Watch!” or “Bas-terd!” Once, a drunk emerged from a
restaurant and stumbled towards Teale brandishing a knife. The actor braced himself, but the man simply wanted “Ser Alliser” to hold the utensil to his throat
for a photo. On a visit to Uluru with his family, the tour guide caught sight of his face and exclaimed, “what are you doing here?” Says Teale, “He just kept
staring at me and I realised it was an Alliser Thorne thing. I said, ‘I’m not going to kill anyone, I’m just an actor’. But it clearly troubled him that this
person who had committed this awful crime had suddenly appeared.” Teale has the grace to admit he enjoys this new level of celebrity.
</item 42>

<item 43>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.

“Take away your feelings for Taylor,” Jason added. “What is your honest opinion on how the NFL is treating celebrities at the games?”

Seemingly beating around the bush, Kelce remarked: “I think it brings a little more to the atmosphere, it brings a little bit more to what you’re watching. But
at the same time, I think...”

Before his younger brother could continue his thought, the Philadelphia Eagles player chimed in: “They’re overdoing it.”

“They’re overdoing it a little bit, for sure,” Kelce agreed. “Especially my situation. I think they’re just trying to have fun with it.”

While sports lovers may not be too pleased with the NFL’s focus on Swift, the football league’s supposed obsession with ongoing Swift and Kelce dating rumours
has paid off in terms of ratings.
</item 43>

<item 44>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Photograph: Adam Vaughan/EPA

They held the ball nicely, probed in neat, clipped, mannered attacks. A lot seemed to hinge on how desperately Bayern felt the need to twist that scalpel,
whether they could really be bothered to take this thing to the wall. United defended well. Their waves of pressing drew waves of applause. Everyone here seemed
to be trying.

But with 58 minutes gone Copenhagen scored and United’s hopes began to slip beneath the waves. Just over 10 minutes later Bayern made it 1-0 in Manchester.
Kingsley Coman got it, granted a huge expanse of time and space in front of the United goal, enough to set himself, as though taking a penalty, and belt the
ball into the corner. The moment was made by a delightfully sneaky round‑the‑corner pass from Kane. After all that pressure, all that tender hope, it looked a
bit like a training goal.
</item 44>

<item 45>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Asks the Queen if she’d received the invitation to Camilla’s 50th birthday, to which she says she has, but cannot attend as she’s in Derbyshire. When Charles
points out that plans “can always be changed”, she replies: “Why would we want to change it?” She doesn’t attend, but later tells Prince Philip that she doesn’t
want for Camilla to be considered “wicked, because she’s not” or to be unkind to her.

Charles (West) hosts Camilla’s (Williams) birthday (Netflix)

It took a while for Queen Elizabeth and Camilla to grow close. According to historian Robert Lacey, she was reluctant to accept Charles’s request to be more
accommodating to Camilla. Unlike in The Crown, she is said to have once referred to Camilla as “that wicked woman” to Charles’s face. However, she grew to
become closer to Camilla.
</item 45>

<item 46>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“After what we saw today, we should all be supporting Britney at this time. Regardless of our past, good and bad, and no matter how long ago it was,” Timberlake
wrote on X, formerly known as Twitter. “What’s happening to her is just not right. No woman should ever be restricted from making decisions about her own body.”

Wade Robson: 2001 to 2002

Shortly after her breakup from Timberlake, Spears was rumoured to be dating Wade Robson, a dancer and choreographer who appeared in the music video for her 2009
tune, “(You Drive Me) Crazy”. However, neither Spears nor Robson confirmed these rumours.

Colin Farrell: 2003

(Getty Images)

Back in 2003, Farrell and Spears made headlines when they were spotted holding hands at the premiere of The Recruit, sparking speculation that they were dating.
</item 46>

<item 47>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”
</item 47>

<item 48>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Justin had embarrassed me publicly, so my rebuttal onstage was to kind of go there a little bit, too. But it was absolutely horrible.”

She has an immaculate sense of shade

Britney seems too pure of heart for active malice, but she has an immaculate way with a piercing characterisation. Timberlake’s band, ‘NSync, were big hip-hop
heads, she writes, and sometimes “tried too hard to fit in” with Black artists. When Timberlake spots Ginuwine at an event, he “got all excited and said, so
loud, ‘Oh yeah fo shiz, fo shiz! Ginuwiiiine! What’s up, homie!’”

Her account of being encouraged to abort their pregnancy at home, lest anyone see them going in and out of hospital, is horrifying. Britney, with no pain
relief, is writhing on the floor. Timberlake proves to be the last person you’d call in a crisis.
</item 48>

<item 49>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
She then poses for them in her swimsuit, but complains in a later episode that they can “never relax” with the press “constantly” around.

Diana (Debicki) on the Fayed yacht (Daniel Escale/Netflix)

Former Vanity Fair editor Tina Brown, who wrote the controversial 2007 book Diana Chronicles, claimed that Diana would often tip off the press and “ couldn’t
resist giving them the images they wanted”, including on the boat. Photographer Daniel Pirrie, who ended up selling the photo of the kiss, claimed that Diana
herself invited him to photograph her on holiday with Dodi in France that summer. “She told me she’d be on the yacht somewhere in the Mediterranean,” he told
The Daily Mail in 2013. “She let me know she was going to be on a tender going out to the yacht. I think overall, she was happy with all the pictures taken that
summer… I knew that Diana had wanted [the kiss photo] to be taken.”
</item 49>

<item 50>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
The supporting cast includes Matt Day (Rake) and Jonathan LaPaglia (The Slap).

Advertisement

Also on Binge: I won’t spoil it, but there was a nerve-shredding cliffhanger in the first season of Vigil (December 11) that I still think about far too often.
That’s a pretty strong indicator that this Scottish crime thriller, where a police detective was assigned to a murder investigation on board an on-duty British
nuclear submarine, pushed the right buttons. The show’s second season stays on dry land but retains the military focus: when an attack drone demonstration at a
British army base goes tragically wrong, Detective Chief Inspector Amy Silva (Suranne Jones, Gentleman Jack) and her now pregnant partner, Detective Sergeant
Kirsten Longacre (Rose Leslie, Game of Thrones), are called in to investigate. Expect a fast pace and significant twists – this series doesn’t do quiet
contemplation.
</item 50>

<item 51>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It may be over a decade since Britney Spears and Kevin Federline ended their marriage, but the former couple continue to have their ups and downs when it comes
to co-parenting their two teenage sons, Sean Preston and Jayden.

Ahead of the release of the pop star’s memoir, The Woman in Me, which is due for release on 24 October, fans are anticipating Spears will share new details
about her relationship with her second husband, Federline.
</item 51>

<item 52>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
In conversation with his brother Jason, 35, the two addressed concerns from longtime football enthusiasts who were frustrated by the constant Swift references
throughout Sunday’s game at MetLife Stadium.

Not only did the football league promote last Sunday’s game by speculating whether the Grammy winner will attend the Chiefs-Jets match, but cameras were
constantly panning to the star-studded suite. During the game, Swift was joined an A-list group of guests, including her best friends Blake Lively and Ryan
Reynolds, Game of Thrones star Sophie Turner, and actor Hugh Jackman.

The NFL also introduced a special segment with Carson Daly and The Voice, where the famed host attempted to explain the specifics of football to “Swifties.”

In the New Heights podcase episode, titled “The NFL Needs to Calm Down”, Jason asked his brother plainly whether he thought the football was “overdoing it” with
the coverage of Swift during the game.

“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.
</item 52>

<item 53>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 53>

<item 54>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
It's amazing to see every day when we all assess the situation, how all these bodies put their egos aside and I'm proud to be a part of it. If any channel wants
an official spokesperson for the Israeli government, then we will be there because we cannot abandon the battlefield."

The world does not make life easy for Israel's spokespeople, to say the least, compared with the ease with which the Hamas narrative is adopted. "I remember an
interview the day after the bluff about the Al-Ahli hospital, which the Palestinians claimed had been hit by an Israeli missile, killing 800 civilians, but in
the end it turned out that it was actually by a splinter group of Islamic Jihad, and that no hospital was destroyed and certainly not 800 people were killed. I
explained to them in Sky News that Hamas lies and makes up numbers and the reporter asks me almost speechless with surprise, if I doubt Hamas' numbers."

Levy describes these moments as intensely frustrating.
</item 54>

<item 55>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
“If the idea of my being in that place was to heal, that was not the effect.”

She’s not scared of anything after that experience, she writes, “but it doesn’t make me feel strong; it makes me sad. I shouldn’t be this strong.”

#FreeBritney protesters in 2021. Photograph: Étienne Laurent/EPA

A nurse shows Britney the blossoming fan movement that is spreading awareness of its suspicions that all is not right with her conservatorship. Come June 2021,
she calls 911 to report her father for conservatorship abuse days before a probate court hearing on the arrangement. “My voice had been used for me and against
me so many times that I was afraid nobody would recognise it now if I spoke freely,” she writes. When she is granted her freedom in November 2021, she says she
feels shock, relief, elation, sadness and joy.
</item 55>

<item 56>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.
</item 56>

<item 57>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 57>

<item 58>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
"I was asked twice, in Ireland and by the Voice of America, why Hamas had been begging for five weeks to return hostages and we were only now bowing to US
pressure to take back our children. I was interviewed by Piers Morgan in London who is very confrontational, who spoke to me about pictures from Gaza and said
'this the most shocking thing I have ever seen.' And I told him Piers, 'the most shocking thing we have seen in our lives was the slaughter on October 7.' Then
he answered 'you are right.'"

From questions like these, do you get the feeling that everyone in the world hates us and they are all anti-Semites?

"It's not true that everyone hates us and they are all anti-Semites. We enjoy a great deal of support in the world. People understand that what happened on
October 7 crossed a line with atrocities that cannot be repeated and for which there can be no excuse.
</item 58>

<item 59>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
STARS WE'VE LOST IN 2023

"Robin's highs were so high and his lows were often low," his "Mork and Mindy" co-star Gina Hecht said during the documentary.

Due to Williams' high-energy performance onstage, many speculated that the comedian had turned to cocaine, with some saying they would see the actor head to the
back of The Comedy Store, where drugs were prevalent.

Williams' longtime friend Allan Stephan recalled a moment that led him to worry.

"He said, ‘Know anybody with any blow? I have to go on. I can’t go on without the blow.’ And I sat down and said, ‘I’m going to help you,'" Stephan claimed. "He
goes, ‘You have blow?’ I go, ‘No, are you out of your f---ing mind? You’re Robin Williams!’

"And then I think after that he wouldn't get high when he had to perform."

However, Williams continued to do drugs after his stand-up shows.
</item 59>

<item 60>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
Robin Williams' sudden death shocked the world in 2014, and now his friends and colleagues are giving an inside look into the years before he died by suicide.

The depth of Williams' drug use was highlighted in Vice's "The Dark Side of Comedy" documentary series. The comedian's episode marked the premiere of the show's
second season.

Williams began his career in stand-up comedy in the 1970s and elevated his fame after appearing in an episode of "Happy Days" as an alien named Mork. The role
led to his own TV series, "Mork and Mindy." The comedian also landed a role in the film world with "Popeye."

Williams' friends noted that while he loved being around people, he also "cherished" his alone time, and they began to see him struggle with his level of fame.
</item 60>

<item 61>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
His character Lars Hutton, an irascible bully and the man most likely to be chucked off the oil rig, has somehow made it back for the next instalment of the
Amazon Prime drama. Hutton, Thorne and Chief Inspector Philip Osborne, another textbook bully who first appeared in the opening season of the BBC police
procedural Line of Duty, are just three of the menacing characters he’s played in recent times. Peter Knox, a villainous sorcerer in A Discovery of Witches, the
fantasy TV series based on books by Deborah Harkness, is another.

Advertisement

When he portrays Ebenezer Scrooge in an upcoming Melbourne production of Jack Thorne’s award-winning adaptation of Dickens’ A Christmas Carol, he’ll summon the
same forces to make the archetypal miser utterly convincing. The truth is this: the avuncular man in the amber-coloured spectacles and leather jacket chatting
to me on the line from Leeds is a master of darkness.
</item 61>

<item 62>
Breaking down each VAR intervention from wild first half of Tottenham vs Chelsea officiated by Michael Oliver (Sporting News, 2023-11-06T22:06:19+00:00)
Chelsea have a THIRD goal disallowed as Sterling is offside 🤯 pic.twitter.com/brMes6qQQ7 — Mail Sport (@MailSport) November 6, 2023

Reece James avoids red card despite elbow to Destiny Udogie

While two Tottenham defenders escaped an early shower earlier in the match for incidents that could have potentially seen them off, Chelsea also had a player
avoid further punishment just before halftime.

Reece James caught Destiny Udogie in the face with a trailing elbow while challenging for a bouncing ball in the air. Referees have been strict with incidents
such as these, and it was surprising to see James get away with just a yellow card.

The incident was indeed reviewed, but the VAR official did not ultimately advise a check for Michael Oliver.
</item 62>

<item 63>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 63>

<item 64>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 64>

<item 65>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 65>

<item 66>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
With the show heading into the late Nineties and early Noughties in its final season, which arrived on Netflix on Thursday (16 November), events are being
depicted on screen that many of the public will remember well.

The first part of season six takes place in the summer of 1997, and sees Prince Charles (Dominic West) building on his relationship with Camilla Parker Bowles
(Olivia Williams), while Princess Diana (Elizabeth Debicki) sparks up a romance with film producer Dodi Fayed (Khalid Abdalla). The show also features the
pair’s death in a car crash in Paris.

As the new season arrives, and fans wonder what in The Crown is based in reality, here’s everything you need to know.

Diana’s role with Tony Blair’s government

Episode one sees Diana visiting Tony (Bertie Carvel) and Cherie Blair (Lydia Leonard) with William after asking for a meeting with him.
</item 66>

<item 67>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
At the same time his look of astonishment and raised eyebrows have become a popular meme and WhatsApp sticker. "If you would have told me several weeks ago that
people would make memes of my face I wouldn't have believed it. But joking aside, we are here to explain the story of Israel abroad, if along the way there are
amusing moments that allow people to enjoy a funny moment amid all the darkness and gloom we are experiencing, then let the joke be on me."

Are you are asked many questions that smack of antisemitism, or lack of understanding and support for Israel's position?

How long have you got? I was asked by the BBC in a radio interview whether there is CCTV evidence that the hostages are in Shifa Hospital and what medical
treatment were they receiving. I was asked by Sky News about why Yahya Sinwar was releasing hostages, even though Israel is bombing Gaza and I had to explain
that it was because Israel is bombing Gaza.
</item 67>

<item 68>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
</item 68>

<item 69>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
However, according to the Reddit user, her husband didn’t take the news as expected. She explained that he asked her a series of questions, with the first one
being if she asked Calla if both her mother and stepfather could walk her down the aisle. The woman noted that she didn’t ask her daughter that.

“Then he asked if she ever considered asking him, and I said I could not answer that for her. He asked me if I thought of him when I said yes,” she continued.
“He asked whether I gave any thought to all he has done for Calla, for both kids, and the fact he’s still not looked upon as a fatherly figure all these years
later. I told him I did not think of him when asked because I was overjoyed. He told me I should have given him a lot more consideration and I should have tried
to compromise with my daughter.”
</item 69>

<item 70>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Adams and McStay – who arrived from the Lions in the off-season as an unrestricted free agent – leaned on each other to cope with the difficult moment, and
needed some privacy to deal with their personal misfortune before they could share in the team’s raucous celebrations. McStay may need as many as eight weeks to
recover from the high-grade medial collateral ligament strain he suffered in the club’s preliminary final win. Collingwood’s injured recruit Dan McStay at
training before the grand final. Credit: Eddie Jim “We shed a few tears and a bit of emotion, then we brushed it off and said, ‘Let’s get on with it and go
celebrate with the boys’. It was a great night,” Adams said. John Noble was another sad story, playing every game in the home-and-away season, only to be
dropped for Collingwood’s qualifying final clash with Melbourne and never forcing his way back in.
</item 70>

<item 71>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
But his pale face, intense eyes and imposing physicality – he’s 188cm in his socks – will be. The Welsh character actor is a Zelig-like figure whose frequent
stage and screen appearances invariably guarantee two things: the tale will be well told and bad stuff is going to happen. He’s in Leeds when we speak, visiting
his younger daughter Gracie, 21, who’s studying art and design at the city’s university. Eliza, 25, his other daughter with actor wife Sylvestra Le Touzel, is
following in her parents’ footsteps. His son, Ion, from his first marriage, recently made him a grandfather. Owen Teale and his wife, Sylvestra Le Touzel, at
the sixth-season premiere of Game of Thrones in Hollywood. Credit: Getty Images Loading Earlier today, he was in Edinburgh shooting scenes for the second season
of the supernatural TV thriller The Rig.
</item 71>

<item 72>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.
</item 72>

<item 73>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Mohamed al Fayed’s role in securing kiss photo

Throughout season six, Mohamed al Fayed (Salim Daw) is shown as heavily invested in his son Dodi’s relationship with Diana, even ringing up the maid to ask if
they have been “intimate”. In Arabic, he then asks his secretary if they can find him an Arabic photographer – “not just any idiot with a long lens”, but “the
best photographer on the Meditarranean”.

Salim Daw as Mohamed al Fayed (Netflix)

It’s not known whether Mohamed was so instrumental in securing the photo of Dodi and Diana kissing for the first time. However, following the Harrods owner’s
death this year, biographer Tom Bower recalled in The Times how Al-Fayed “gleefully” showed him the photographs of Dodi and Diana and allegedly “revealed how he
had rapidly bought a yacht suitable for his son’s seduction of her”.
</item 73>

<item 74>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"The concentration of ketamine found in Matthew Perry’s blood was sufficient to cause loss of consciousness and lack of responsiveness to external stimulation,"
Lewis Nelson, chair of the Department of Emergency Medicine and chief of medical toxicology at Rutgers New Jersey Medical School, said in a statement provided
to Fox News Digital.

"This explains why he slipped under the water and did not awaken. Ketamine does not generally cause death from cardiac or respiratory effects, but rather from
associated injury."

MATTHEW PERRY CAUSE OF DEATH LISTED AS 'ACUTE EFFECTS OF KETAMINE'

Dr. Bankole Johnson, CEO and founder of Casa Privée in Miami , a concierge medical facility, offers ketamine infusion therapy to help patients manage various
conditions.

He told Fox News Digital that in his opinion, Perry’s death likely was linked to recreational ketamine use, although he did not treat or examine the actor.

Experts shared with Fox News Digital the following common myths and misconceptions about the drug.

Myth No.
</item 74>

<item 75>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
Williams reportedly appeared tired while on set and didn't seem to be his usual self before he died by suicide on Aug. 11, 2014.

The comedian was diagnosed with the neurological disorder posthumously after an autopsy was performed. Lewy body dementia is the second most common type of
dementia after Alzheimer's. The disease affects brain regions involved in thinking, memory and movement, according to the Mayo Clinic.

The autopsy found no illegal drugs or traces of alcohol in Williams' system.

"The last time I saw Robin Williams, I hadn’t seen him for a few months, and he was thin, and he didn’t recognize my wife," Steven Pearl, a comedian and friend,
said during the documentary. "He hardly said a word. It took him a second to recognize me. I go, ‘Whoa, something’s wrong here.’ I didn't know what. Nobody
would really tell me. And then I never saw him again after that."

CLICK HERE TO GET THE FOX NEWS APP
</item 75>

<item 76>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
At the time, her representative also confirmed to People: “Yes, she’s single.”

The news of the breakup came asTMZ alleged that Lucado had cheated on Spears, with claims that her father had purchased a video of the infidelity incident. The
publication later alleged that Lucado denied the cheating allegations, with claims that he was seeing the other woman while he and Spears were broken up.

Despite their seemingly rocky history, Lucado showed his support for his ex in a 2016 interview with theNew York Times about her conservatorship. Speaking to
the publication, he said: “If anyone knew the real Britney, they would know that she would rather be remembered for being the great mother she is rather than
the artist she is.”

The publication also noted that Lucado denied the infidelity claims from Spears’ father, before he added: “And if anyone could see her interactions with her
kids, they would know that there is no need for a conservatorship over Britney’s personal life.”
</item 76>

<item 77>
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
</item 77>

<item 78>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 78>

<item 79>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 79>

<item 80>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Dodi’s proposal to Diana on the night they died

After bursting into tears going for dinner at the Ritz, move upstairs, he proposes to her with a ring she had previously pointed out when they sheltered from
crowds in a Monte Carlo jewellery shop. She tells him to get up off his knee “I know the whole world is wondering if we’re going to get married, but that’s not
a reason to actually do it.” However, they share a drink and bittersweet conversation about how they need to live their lives going forward - advice

Dodi is shown proposing to Diana before they died (Daniel Escale/Netflix)

In real life, Dodi did not propose to Diana on that fateful evening in Paris. However, at the inquest into the death in 2007, the jury were shown CCTV footage
of him purchasing an engagement ring worth £11,600 in a jewellers across the square from the Ritz on the afternoon of the crash.
</item 80>

<item 81>
Bride defended after asking mother to walk her down the aisle instead of stepfather (The Independent - Life and Style, 2023-11-27T18:21:04+00:00)
She specified that she also has a husband, who’s been a stepfather to her daughter since she was 10.

The parent went on to give more context about her family, noting that she lost the father of her children - Calla and Hawthorn - when they were six and eight.
She then shared that she remarried four years later, before opening up about her relationship with her second husband and how her children reacted to it.

“We were friends for several months, dated some, stronger feelings developed and I introduced him to the kids to see how they would get along,” she wrote. “We
halted for a year while my kids and I did some therapy because their reaction to my husband was strongly negative, because they didn’t want to replace their
dad. But once therapy was ongoing, they were doing better. From there, things move faster but the kids were on board for things to move on.”
</item 81>

<item 82>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
While celebrities like Madonna, Paris Hilton, and Selena Gomez attended the festivities, Spears’ close family – her mother Lynne, father Jamie, sister
Jamie-Lynn, and her two sons – were not present at the nuptials.

Although their marriage last only 14 months, with Asghari filing for divorce in August this year, the actor and former dancer was previously praised by fans
over his continuing support for Spears amid her conservatorship and custody battle.

In August 2022, Federline claimed during an interview with ITV that their teenage sons found it “tough” seeing their mother sharing nude photos online, saying:
“I can’t imagine how it feels to be a teenager having to go to high school.”

Spears hit back at her ex-husband in a statement shared to her Instagram Stories responding to Federline’s “hurtful” comments.
</item 82>

<item 83>
Author Melissa Lucashenko on playing with black and white binaries (The Age, 2023-10-06T04:50:00+00:00)
But now there’s so much about us in the media all the time, and in the education system to an extent, most Australians have to recognise that Aboriginal people
haven’t died out.” Loading (Census data shows the number of Aboriginal and Torres Strait Islander people has in fact risen – from 548,368 in 2011, to 649,171 in
2016, then up 25 per cent to 812,728 in 2021. This increase is attributed in large part to people feeling more comfortable identifying as Indigenous, and also
to the growing popularity of genealogical research.) That Aboriginal people were not all killed off in the early days of settlement was despite the best efforts
of some colonists; one of the historical events that inspired the book was the public hanging of the Indigenous warrior Dundalli, the last person publicly
executed in Queensland.
</item 83>

<item 84>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Today I miss the days when we were on the verge of a civil war because of a gender partition (in Dizengoff Square)."

"It's so ridiculous what we argued about and what we pulled our hair out over for before October 7, a date that reminds us all of the big picture, why we are
here, why we owe each other and in the end also love each other. It's sad that it took such a tragedy to remind us of the importance of unity in the nation, but
we must maintain it."

Published by Globes, Israel business news - en.globes.co.il - on December 3, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 84>

<item 85>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
Through tears, she claimed that he had proposed to her (something his family denied) and promised to buy her a house in Malibu. However, she dropped the lawsuit
following Dodi’s death.

Diana’s close relationship with paparazzi on France yacht trip

When Dodi and Diana first spend time together on his father’s yacht in France, the paparazzi are never far away. However, when William refuses to go outside
until the photographers leave them alone, she takes a boat out to visit them. Wearing a swimming costume, she asks them: “We’re having a lovely time, apart from
one little thing: you lot.” “Don’t be like that, you love us really,” they reply. Diana asks them to leave her alone as they’re “freaking out” her sons, before
telling one photographer – who she knows by name – that if they go away, “you’re going to get a big surprise with the next thing I do”.
</item 85>

<item 86>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.
</item 86>

<item 87>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
After a while, I’d get bored, sit on the big wheel and go round and round.” The sight of a disconsolate, sweaty bear riding the ferris wheel caught management’s
attention and Teale was sacked. At which point a couple of his co-workers – students from the Guildford School of Acting – suggested he audition at their alma
mater. He was accepted, awarded a study grant and Barry the Bear got a ticket out of Wales.

Advertisement

“I was going nowhere really. If I hadn’t met those two young women I don’t know what would have happened,” says Teale. “But there have been so many points in my
career, moments of pure chance, when things went my way. And if you can survive the moments when things don’t go your way, you can make it.” If you’re not a
diehard Game of Thrones fan, Teale’s name may not be familiar.
</item 87>

<item 88>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“It was heavy and very manic, and incredibly invasive. And it had a kind of pressure to it,” Debicki said in an interview with Netflix.

“At times it’s almost like an animalistic response to being pursued, by that many actors playing the press, because there’s nowhere you can go and you only have
to be in a situation like that for about a minute, before you realise this is completely unbearable.”

The first four episodes of The Crown season six are available to stream on Netflix in the US and UK.
</item 88>

<item 89>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
“I feel as though Will has a tremendous love and commitment to Jada, as well as to the marriage and the family unit,” she said. “The pride that he takes in
defending his wife is extraordinary, almost to the detriment of his image, at times.”

Ultimately, the public will never fully know what “heavy-duty work” Will and Jada are doing in their relationship, and according to Winter, it’s not necessarily
any of our business. Still, she believes that their choice to separate is something fluid and outside of the traditional marriage design - an evolving model
that could also come to serve other couples. If Will and Jada have been spending seven years apart as a married couple, why would they decide to go back to
living in the same household?

“They’re in the rough with a machete, carving a new path for themselves,” Winter said. “Why would they say: ‘Hey, let’s go back to the house and just relax?’
</item 89>

<item 90>
He killed Jon Snow. Now Owen Teale is coming to ruin your Christmas (The Sydney Morning Herald, 2023-10-20T12:22:00+00:00)
But it never came my way. My cut of my jib wasn’t quite right. It’s about what your face does on screen I guess.”

Advertisement

There have been opportunities to step into the spotlight over the years. When The Fifteen Streets attracted a massive TV audience in Britain, Teale was offered
a lead role in a movie playing a bare knuckle fighter. But the financing fell through and he signed up for a four-year stint with the Royal Shakespeare Company
instead. Years later, a producer offered him the lead in a big film being shot in Britain, but the money men demanded an American star and William Hurt got the
gig. When Teale won a Tony Award in 1997 for his performance in Ibsen’s A Doll’s House, he was told he needed to spend time in Hollywood wooing producers. He
decided to fly home to his family in London instead. “My wife was about to give birth, so I said ‘I’m off’.
</item 90>

<item 91>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
That same day, the fitness instructor officially filed for divorce, listing the date of separation as 28 July 2023 and citing "irreconcilable differences" as
the reason for the split.

In a statement posted online, Asghari said: “After six years of love and commitment to each other my wife and I have decided to end our journey together. We
will hold onto the love and respect we have for each other and I wish her the best always. S*** happens. Asking for privacy seems ridiculous so I will just ask
for everyone including the media to be kind and thoughtful.”

Days before Spears’ memoir release date, Asghari revealed that he had obtained an early copy of the book. In an interview with TMZ on 13 October, he explained
that he wasn’t worried about anything being written in the book that could paint him in a negative light.

“I already read it and I’m very proud of her,” he told the outlet.
</item 91>

<item 92>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
The model later confronts Dodi, asking if he was with Diana when he leaves her to go see his father. Diana also alludes to a “lawsuit” between Dodi and his ex.

Diana (Debicki) and Dodi (Abdalla) (PA)

This storyline comes to close to the truth as told by model Kelly Fisher, who claimed to have been engaged to Dodi when he met Diana. Fisher had previously
modelled for brands such as Victoria’s Secret and magazines including Elle and Marie Claire. As in The Crown, Fisher – who met Fayed in July 1996 – was on
another Fayed yacht when Dodi and Diana were first photographed. She staged a press conference where she announced that she was filing a breach-of-contract
lawsuit against Dodi, after he allegedly asked her to stop modelling as much and prioritise the relationship.
</item 92>

<item 93>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
“The last call between Harry, William and Diana is heartbreaking,” a third user commented.

In both the series and in real-life, Diana and Dodi - along with their chauffeur, Henri Paul - were killed on 31 August 1997 when their Mercedes-Benz was
pursued by paparazzi and crashed in the Pont de l’Alma tunnel in Paris. The mother of two was 36 years old.

The fatal car crash was recreated in the very first scene of season six. While the incident is not shown on-screen, a car carrying Diana and Dodi is seen
speeding past a man walking his dog, before entering the tunnel. As the vehicle’s horn blares, he calls emergency services on his cell phone.

Australian actor Elizabeth Debicki, who took over the role of Diana from Emma Corrin in season five, admitted it was “difficult to recreate” the scene -
especially with her character being hounded by the paparazzi. “It was heavy and very manic, and incredibly invasive.
</item 93>

<item 94>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
The season begins with Princess Diana (played by Elizabeth Debicki) whisking her sons - Prince William (played by Rufus Kampa) and Prince Harry (played by Fflyn
Edwards) - away on summer holiday to Saint-Tropez, France. After staying several days on Mohamed Al Fayed’s yacht, the boys return home to London where their
father, the then-Prince of Wales, accompanies them to Balmoral Castle to vacation with the rest of the royal family in Scotland.

In episode three, Diana’s partner Dodi Fayed (played by Khalid Abdalla) surprises her with a quick trip to Paris, France. However, the princess is keen to
return home to the United Kingdom, and is eager to speak to her sons over the phone. A mob of paparazzi following the couple delays Diana’s scheduled phone call
with William and Harry, until she’s able to catch them again later that evening.

It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.
</item 94>

<item 95>
Robin Williams' drug addiction made him 'a monster': Friends recall depth of actor's cocaine use (FOX News - Entertainment, 2023-10-19T17:22:01+00:00)
Belushi's death, along with the birth of Williams' first son in 1983, really helped the comedian clean up his drug addiction, according to the documentary.

Williams quit cocaine "cold turkey," his longtime friend Stanley Wilson said.

"He knew what an amazing instrument that he had, and that it is all related to his mind and his imagination and his wit," Wilson added. "The more stuff you do
to deaden that, the more you’re going to burn some more brain cells. Robin didn’t want to do that."

CLICK HERE TO SIGN UP FOR THE ENTERTAINMENT NEWSLETTER

Once he quit, Williams' career continued to grow — with him landing a show at The Met and other movie roles, including "Mrs. Doubtfire," "Good Will Hunting" and
"Night at the Museum."

However, in the years before his death in 2014, the actor was unaware he was dealing with Lewy body dementia.
</item 95>

<item 96>
Theatre producer and Everton chairman Kenwright dies (BBC News - Entertainment & Arts, 2023-10-24T16:32:09+00:00)
Among his current shows, Sir Ian McKellen is starring in Frank and Percy in London, while Twelve Angry Men, Calendar Girls The Musical, Heathers The Musical and
Blood Brothers are all on tour.

Dame Judi Dench, Woody Harrelson, Billie Piper, Rob Lowe and Felicity Kendal have also starred in his productions.

Rooney leads tributes to a 'big inspiration'

Wayne Rooney, former Everton and England forward: "Devastated to hear the sad news about Bill Kenwright. Known Bill since I was young and he's had a huge impact
on me as a person and my career. Great man and a big inspiration. Thoughts are with all Bill's family and friends."

Jamie Carragher, ex-Liverpool and England defender: "Really sad news this. A huge Evertonian who served and loved his club to bits. I'll never forget his and
Everton's support every year around the Hillsborough memorial. RIP Bill."
</item 96>

<item 97>
The Crown: What is fact and what is fiction on season 6? (The Independent - Life and Style, 2023-11-17T19:32:16+00:00)
He is not located, but later walks back to the house on his own accord, drenched in rain. “14 hours, that poor boy was gone,” the Queen later says.

William (Kampa) goes missing after his mother’s death (Keith Bernstein)

It’s true that William and Harry initially stayed in Balmoral with their grandmother following the death of Diana. While it’s not known whether he went missing,
during a visit to Scotland in 2021, William recalled how that experience was one of the “saddest” moments of his life. “I was in Balmoral when I was told that
my mother had died. Still in shock, I found sanctuary in the service at Crathie Kirk that very morning,” he recalled. “And in the dark days of grief that
followed, I found comfort and solace in the Scottish outdoors. As a result, the connection I feel to Scotland will forever run deep.”

The Crown season six part one is on Netflix now, with part two arriving on 14 December.
</item 97>

<item 98>
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 98>

<item 99>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“It’s hard... the thing Will and I talk about a lot is the journey. We started in this [relationship] at a very young age, 22 years old... I think you expect
your partner to know - especially when it comes to sex. It’s like: ‘Well, if you love me, you should know. If you love me, you should read my mind.’ That’s a
huge pitfall,” she said at the time.

The next month, Will’s memoir is released which details several issues the couple have had over the years. He wrote they were “suffering the brutal death of our
romantic fantasies, the burning away of the idealistic illusion of the perfect marriage and the perfect family. Neither of us wanted a divorce; we knew we loved
each other, and some aspects of our union were magical. But the structure of the life that we had established was strangling both of us.”
</item 99>

<item 100>
Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo (FOX News - Entertainment, 2023-12-17T10:00:59+00:00)
If Tasha was here, someone could share this.' But yeah, we’re doing all right, you know?"

He said, "The thing Tasha and I really wanted to instill in them was manners. That may sound like such a basic thing, but I’ve heard so many adults say, 'Oh, my
God, your sons — they look you straight in the eye. They’re very polite and mannered.' It makes me proud. And it can get you through a lot, in a way. Just
showing respect for your fellow man."

LIAM NEESON'S ‘RETRIBUTION’ ACTION HERO ROLE LATEST IN STORIED CAREER MARKED BY LOVE AND LOSS

Neeson also admitted to "drinking too much" after losing his wife. In another 2014 interview, this one with "60 Minutes," he said that grief sometimes hit him
"like a wave. You just get this profound feeling of instability."
</item 100>
=== END ITEM id=aa5c6e7ab6d2494b ===

=== ITEM id=fe0a49b3b09d48f5 ===
Gold sentences and facts:
Sentence 0: While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s.
  Fact 0.0: Altman is not trained as an AI engineer.
  Fact 0.1: Altman is now 38 years old.
  Fact 0.2: Altman has been seen as a Silicon Valley wunderkind since his early 20s.
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture.
  Fact 1.1: The Information reported that Altman has been telling investors about a new venture.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board's language.
  Fact 2.1: The prevailing theory floating around right now is based on the way these giant tech companies work.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 1>

<item 2>
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
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 4>

<item 5>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
“Sam acted in good faith and took reasonable business measures” is a pretty hard pill to swallow with that in mind.

Man, it’s no good when your defense lawyer has just made you sound worse than the prosecution already did. And while Cohen tried to make the common white-collar
defense argument that Bankman-Fried, as CEO, was simply too busy to oversee what everyone did every day, he just made me more suspicious. That’s why you hire a
risk officer and delegate! That’s the whole point! I could barely even hear Cohen blaming Caroline Ellison and Changpeng “CZ” Zhao for the debacle over the “no
risk officer” ringing in my ears.

Following the defense’s opening statements, things got still worse for Bankman-Fried. The prosecution called its first witness, Marc-Antoine Julliard, whose
money got stuck on FTX. Juilliard, who was born in Paris and lives in London, testified that he trusted FTX because Bankman-Fried came across as a leading
figure of the industry.
</item 5>

<item 6>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 7>

<item 8>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=fe0a49b3b09d48f5 ===
