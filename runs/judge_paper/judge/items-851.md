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

=== ITEM id=a1cd67c51ac320e2 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:
<item 1>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
I think they can sense it again. That’s what I think. I think they sense we are getting back to an earthiness, a determined Everton group of players who will
really have a go.”

Dyche declined to comment on the implications of Everton’s punishment for Manchester City, who have been charged with 115 financial breaches by the Premier
League, and deny any wrongdoing, and Chelsea, who are also under investigation by the football authorities.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Emma has endorsed Twila, she will be a key part of Emma’s staff when she arrives and moving forward, and we are excited for what’s to come with our USWNT
program.”

GO DEEPER How Emma Hayes' winning ways at Chelsea can benefit USWNT on the field

It’s still an extremely tight turn for the Olympics, with 18 days for Hayes to get situated with the team ahead of the tournament, between the two international
windows from May 27 to June 4, then July 8 to 16. The Olympic tournament will start on July 25.

Advertisement

There are, of course, logistical questions about roster selection over the next few months. Some of those may be answered relatively soon, with the roster for
the upcoming camp that begins at the end of the month imminent. The greater challenge will likely be ongoing player evaluation over the next six months, at a
time of great transition within the squad. The specifics of how that will work without Hayes’ involvement remain a mystery.
</item 2>

<item 3>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
That would have been too simple. Rather for their next game at Burnley on 7 October when, according to Pochettino, James was back in contention for selection.

The 23-year-old was not called up by England for the internationals over this past window and the idea was that he would work to be ready for Saturday’s home
derby against Arsenal. And yet there have been stories of an injury setback, even that he may need surgery to get to the root of his hamstring issues.

Pochettino bustled his way through those questions on Friday and the best read on James’s status was that he would be back in the squad for Arsenal, most likely
as a substitute. Gusto is available again after his own suspension. “It [James’s hamstring] is much better,” Pochettino said. “He is progressing well, still not
at his best, 100%. But maybe to help the team at some point …. maybe, he is close to be there.”
</item 3>

<item 4>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“We’ll see what the appeal brings. The mentality is key for me, we took a knock but there have been a lot of knocks here over the last couple of years.

Advertisement

“But the team mentality is growing. I can’t emphasise it enough, the commitment to each other and the connection is a powerful thing because I know we’ve got
quality.”

Eddie Howe’s side are proving one of the most inconsistent in the league, albeit with a lengthy injury list.

In the last month, they have defeated Manchester United, Arsenal and Chelsea – as well as drawing at French heavyweights PSG in the Champions League – but have
now lost comfortably to two cellar dwellers in Everton and Bournemouth.

“We have to do better,” said Howe.

“There are reasons behind every performance and we have struggled to change players. That is a big miss with the quality of players out. The longer you do it,
the harder it gets.
</item 4>

<item 5>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
Sarina Wiegman talks a lot about defending against counterattacks, because this is basically the key to the whole enterprise. Cut off the counter, and with
England’s technical quality they can easily dominate. It is also the essential measure of how the team is functioning as a collective: the efficiency of the
press, the closing of the angles, the command of space. Lineth Beerensteyn’s two goals were the sort England simply were not conceding four months ago.

Mary Earps shows her frustration after conceding a second goal to the Netherlands. Photograph: Tom Jenkins/The Guardian

So England passed the ball around a bit and gave it away a bit and essentially ran around trying to feel normal again, and on countless occasions Kelly or Hemp
would win a first or second ball, look up and see a penalty area empty of white shirts.
</item 5>

<item 6>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
However, it's also possible he decides to stay on for the 2026 World Cup cycle, particularly if England have a good tournament in Germany. In that instance, he
would only be available for double duty on a temporary basis, and that would simply be too messy to outweigh the benefits. Brazil are currently in that boat,
and it's all gone wrong for Fernando Diniz despite his Copa Libertadores victory with Fluminense.

Mauricio Pochettino

In the event Todd Boehly proves to sport just as itchy a trigger finger as his successor Roman Abramovich at Chelsea, it's possible that Mauricio Pochettino
becomes available should Manchester United look to move on from Erik ten Hag. But would he really be a sellable hire, having just failed in London after bombing
at PSG and departing Tottenham in acrimonious manner? It seems unlikely that United would be interested in someone who demands significant club control and will
have just been fired from a struggling league rival.
</item 6>

<item 7>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 7>

<item 8>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
This began the race to confirm Hayes had been selected as the USWNT head coach.

Talks between U.S. Soccer and Hayes’ representatives continued even after Chelsea’s press release. The federation’s board convened late on Saturday, Nov. 4 to
approve the selection, even without the final details of the contract settled or signed.

At the end of it all, the sides have agreed to a deal that will make Hayes the highest-paid women’s football coach in the world — though her salary is not tied
to equal compensation with Berhalter. While her salary is in the same range as the USMNT head coach, it’s thought to be a reflection of the market value for
Hayes. With reports that Chelsea was prepared to quadruple her salary to keep her, Hayes herself danced around the details in her first media availability with
the club.

“I believe in private conversations,” she said. “Of course, I’m disappointed to hear things being said in the press.
</item 9>

<item 10>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 10>

<item 11>
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
</item 11>

<item 12>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 12>

<item 13>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Sporting director Matt Crocker made the final decision to hire Hayes after a search process that began in August, following the team’s surprising exit in the
round of 16 at the World Cup and the subsequent departure of head coach Vlatko Andonovski.

“She has tremendous energy and an insatiable will to win,” U.S. Soccer president Cindy Parlow Cone said in a statement. “Her experience in the USA, her
understanding of our soccer landscape and her appreciation of what it means to coach this team makes her a natural fit for this role and we could not be more
pleased to have her leading our women’s national team forward.”

Though Hayes is seen as one of the world’s top coaches in women’s soccer, the appointment still comes as something of a surprise. Here’s how the deal got done.

Details of the deal

At least part of the surprise surrounding Hayes’ hire – and the six-month runway before she officially takes charge – is down to U.S. Soccer’s own messaging.
</item 13>

<item 14>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 14>

<item 15>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“This is a huge honor to be given the opportunity to coach the most incredible team in world football history,” she said in a statement on Tuesday. “The
feelings and connection I have for this team and for this country run deep. I’ve dreamed about coaching the USA for a long time so to get this opportunity is a
dream come true. I know there is work to do to achieve our goals of winning consistently at the highest levels. To get there, it will require dedication,
devotion and collaboration from the players, staff and everyone at the U.S. Soccer Federation.”

Looking ahead for the USWNT

Hayes’ appointment will have an immediate impact, even if she’s not immediately present. The particulars around timing and the plan moving forward have been one
of the areas of discussion between Hayes and her representatives, Chelsea and U.S. Soccer that has continued through the start of November.
</item 15>

<item 16>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Only leaving the side to reach the pinnacle of his job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed
just how big picture he is with an interview in 2021.

“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way we travel, the way we
refuel.

“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”

Having worked in football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made
often only paying off after he’s left.

One in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.

“I took quite a lot of criticism in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.
</item 16>

<item 17>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 17>

<item 18>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
The 44-year-old Italian has led the tiny club to new heights, qualifying for the Europa League for the first time in club history and establishing themselves as
top-four contenders, building upon the success under Graham Potter. Yet could Potter's failure at Chelsea cause some trepidation around another Brighton success
story moving on up?

Michael Carrick

This list wouldn't be complete without mention of the likeliest of those with direct Manchester United links. Former United midfielder Michael Carrick, who made
316 appearances for the club over a 12-year Old Trafford tenure, is at Championship club Middlesbrough, his first permanent managerial appointment. They sit
11th in the second tier as of mid-December, just three points back of the playoff spots in a crowded top half.

Carrick's lack of experience is glaring, but hiring someone with deep knowledge of the club could be very enticing, and the fact he hasn't bombed in his first
job is notable.
</item 18>

<item 19>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Even if you’re not playing well, if the team is not playing well, or you’re not playing to the standards you expect, you still have to win – so find a way to
win, be pragmatic.”

United have been so of late, claiming victories over Sheffield United, Brentford and FC Copenhagen in their last three matches. But Ten Hag’s “depends on the
opposition” and “how good is the team around him” comments sound like hints. In contrast to Inter’s Simone Inzaghi, the United manager may harbour doubts about
deploying Onana high up the pitch against City.

The Dutchman is still building his team and may be wary of the uneven form of United’s new keeper. Bought for £47.2m, Onana arrived on a ticket of being a
high-grade shot-stopper, technically impressive too with the ball at his feet. But until impressing against Brentford and Copenhagen, Onana had bombed, costing
United goals and points.
</item 19>

<item 20>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
I’ve never lacked confidence since I’ve been here and, let’s face it, I’ve had some pretty tough times. But it’s not just about confidence. You’ve got to win
games, and this is just another big challenge.

“It is currently a momentary thing because we don’t know what the appeal brings, we don’t know how that changes things again. But until that period, and I have
made it clear to the players, be under no illusions because there are no guarantees things will change so we have to commit to what’s going on now as if that is
the reality. It might be changed but, until it’s changed, we have to commit to this reality. There are no guarantees that what goes before guarantees what comes
next. We have to keep pushing like we have been doing because we have lost 10 points and we have to get them back as quick as we can.”
</item 20>

<item 21>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 21>

<item 22>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 22>

<item 23>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
The 22 year-old first turned in Connor Gallagaher’s 94th minute pass followed by a clinical strike from Palmer’s pass in the 97th minute, which secured Jackson
his first Premier League hat-trick of his career.

Postecoglou meanwhile suffered a first home defeat as a manager since he began his 52-match unbeaten home run in November 2020 during his time at Yokohama F.
Marinos, then two-year spell at Celtic plus now Tottenham.

What’s Next?

Tottenham continue their push for the Premier League title with a trip to Wolves on Saturday 11 November, with kick-off at Molineux set for 12:30pm GMT live on
TNT Sport 1 in the UK.

Chelsea host Spurs’ title rivals, Manchester City on Sunday 12 November at 4:30pm at Stamford Bridge, live on Sky Sports in the UK.

Team News

Tottenham: Guglielmo Vicario, Pedro Porro, Cristian Romeero,
</item 23>

<item 24>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
It was the moment when Reece James simply could not handle the frustration. Named by Mauricio Pochettino as the Chelsea captain, James had not played since the
opening weekend draw with Liverpool at Stamford Bridge when he injured a hamstring. Now, he seethed from the sideline as his team lost 1-0 at home to Aston
Villa, a game in which his replacement, Malo Gusto, was controversially sent off.

It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again. James would
lose his cool with an official in the tunnel and, when a Football Association charge was announced three days later for abusive conduct, it felt like only a
matter of time before he was given a one-match ban.

It did not come in time for Chelsea’s visit to Fulham on 2 October, when James was still injured. That would have been too simple.
</item 24>

<item 25>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Speaking to BBC post-match, Tottenham manager, Ange Postecoglou said: “There was a fair bit going on, a lot of VAR intervention, a hard game for me to analyse
but I am very proud of the players. Right to the end they were trying to create something for us.

“The players couldn’t give any more than they did, part of this game is will, effort and determination and they showed that.”

Tottenham now sit second in the table, whilst Chelsea’s win lifted them up to tenth on goal difference from fellow London rivals, Crystal Palace.

Embed from Getty Images

Having slipped to second in the table over the weekend, Tottenham started with high energy and fluidity which saw Pape Matar Sarr play Son Heung-min through in
the fifth minute, but the South Korea international saw his shot saved by Robert Sanchez.
</item 25>

<item 26>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Bruno is an [effective] player when we have a lot of the ball. Then he is at his best. And there are games against top teams when you have less of the ball. So
then it is about taking benefit from [any attacking] moments and for the rest you have to work hard for the team. And to get a result because that is the most
important, that the team is winning.”

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply.
</item 26>

<item 27>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 27>

<item 28>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
They’ll always have Paris. Throughout this Nations League campaign this has been an England team fighting its own sense of entropy, still giving everything,
still trying the same plans and processes in the hope that eventually something would click. And at 9.34pm on a briskly cold Wembley night, it finally did. Ella
Toone’s injury-time goal earned England three points and kept their faintly glowing Olympic dream alive. But somehow it also did so much more.

It breathed life into a team and a cycle, and a moment that for so much of the evening had felt like it was drawing to its natural conclusion provided a rousing
encore to a cast preparing for its final curtain. There will be new challenges and new frontiers to conquer soon enough, a fresh set of European Championship
qualifiers in the spring, but for now this iteration of the Lionesses still has tricks up its sleeve, still has talent and creativity to burn, still has a fire
that can scorch any team on the planet.
</item 28>

<item 29>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Palmer eventually stepped up in the 35th minute and Vicario got a touch on the Englishman’s penalty, yet could only deflect it onto the right post, which saw
the ball ricochet into the bottom-right corner to pull the West London visitors level.

Chelsea quickly pushed on after their equaliser against an imploding Spurs defence but failed to find a breakthrough goal before half-time, whilst the hosts
lost Romero’s central defensive teammate – Mickey van de Ven to injury alongside James Maddison in two huge blows.

Tottenham however were lucky to not be down to nine men at the break after Sarr and Colwill came to blows in the tenth minute of injury time, yet Oliver decided
to just yellow card both players than send either player off.

Reece James was then lucky not to be dismissed for Chelsea in the 11th minute of injury-time after he elbowed Udogie in the face, yet VAR deemed it a simple
incident and not a yellow or red card offence.
</item 29>

<item 30>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
What is the style of play, what’s the plan?

“He needs to take a step back, ten Hag, and say what do I need to do with these players I have available in order to first of all stop leaking goals, because
that’s a problem, and secondly, score goals – how do I get the next goal? Four games without a goal, it’s a long time.”

Advertisement

DEADLOCK BROKEN! West Ham lead against Manchester United! ????



That assist from Lucas Paquetá though… ????



Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/Qqq7zugQmj — Optus Sport (@OptusSport) December 23, 2023

Ten Hag tried to put a positive spin on the performance by claiming to BBC that they were the better side before the first goal.
</item 30>

<item 31>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Arsenal represent an acid test of the progress and the start of a daunting run in the league; after them, Chelsea face Brentford, Tottenham, Manchester City,
Newcastle, Brighton and Manchester United. Pochettino, though, pushed back against the notion that he was seeking a first statement win at the club, something
to ignite his tenure. All he wants are points to drive his team up from 11th in the table.

Pochettino has appeared mired in a selection crisis since his arrival at Chelsea; James only one element of it and he will be without seven others through
injury against Arsenal.

“The moment we have all the squad at a high level and fit, we can say: ‘OK, we are going to Arsenal, Manchester City and we are going to beat you.’ At the
moment, with the circumstances, I don’t think this game is going to be a statement to say: ‘Be careful, Chelsea is coming,’ or something like this.”
</item 31>

<item 32>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 32>

<item 33>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 33>

<item 34>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 34>

<item 35>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“It’s how you are and about your character,” Pochettino said. “He feels football. I am 51 and still sometimes I feel that decisions are no good or unfair. You
are competitive. Like with Mikel or myself or other coaches that played football, in our minds, we still believe we can play. Still, we have the animal
instinct. We cannot be thinkers.”

For Pochettino, it must be about channeling the emotion in the right way and not suppressing it. More broadly, he is in a better place after the wins at Fulham
and Burnley, which followed the Carabao Cup victory over Brighton on the day that James was charged.
</item 35>

<item 36>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 36>

<item 37>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer that has continued through the start of November.

Unless the situation changes drastically, Hayes will only have two camps, including four friendlies, with the USWNT ahead of the 2024 Olympics in Paris. She’ll
miss three international windows between Tuesday’s announcement and her planned start date, including the 2024 CONCACAF Gold Cup in February and March.

U.S. Soccer has a plan in place for the transition. Current interim head coach Twila Kilgore will continue in the role through May and will remain as a
permanent assistant coach under Hayes after the swap is complete. U.S. Soccer said that Kilgore and the rest of the technical staff are working on a handoff
plan for Hayes.

“This is a unique situation, but the team is in safe hands with Twila,” Crocker said. “Her stewardship will be crucial during this period as we are focused on
success at the Olympics.
</item 37>

<item 38>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s own messaging. Crocker, in a September meeting with U.S. reporters along with Cone and U.S. Soccer CEO J.T. Batson, said he hoped to have a new head
coach in place by December.

But the initial contact with Hayes was made a couple of months ago, early into the search, with all three top-level executives from U.S. Soccer involved in
those talks. The trio also described the interview process to journalists in that September meeting – a U.S. soccer statement describes it as involving
“psychometrics and abstract reasoning tests, in-depth discussions of strategy, coaching philosophy and the current player pool, as well as evaluation on the
reactions to pressure, culture-building and interactions with players and staff.” USMNT coach Gregg Berhalter went through a similar process, including an
abstract reasoning test, when he was re-hired by Crocker earlier in 2023.
</item 38>

<item 39>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 39>

<item 40>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
It is a global context." Does he still feel the same way?

Graham Potter

While Potter's time at Chelsea crashed and burned, many still rate the former Brighton & Hove Albion boss highly and he could be in line for a project job such
as this. Indeed, there have previously been rumours that Sir Jim Ratcliffe, the INEOS founder on the brink of purchasing a 25 per cent stake in United, would
like Potter at the club. He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with
the London club went so poorly.

Roberto De Zerbi

One of the fastest rising managers in the game, Brighton manager Roberto De Zerbi is known for his tactical innovation in transitional play, and his name has
cropped up a number of times when big clubs have an opening.
</item 40>

<item 41>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
</item 41>

<item 42>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
"We are a long time from the Sir Alex Ferguson era. Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five
years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right
things.

"You have to know that, over that period, the consistency of the decision-making will yield a good end result. I think we've seen that both Manchester City and
Liverpool, that when you adopt that long-term strategy you can get real consistency and sustainable success.

"I think that's the biggest thing for Manchester United; putting down their identity of what they want to be today, but most crucially what they want to be in
five years' time, and where they want to be, and what they want to look like; from everything from young players, to the style of play, to the whole culture
peace in Manchester United."
</item 42>

<item 43>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
They still have the FA Cup, and the extended pursuit of fourth or even fifth place, although the state of the coefficient means the thing that might keep
Manchester United out of Europe next season could be Manchester United being so bad in Europe this season. Maybe City can revive their fortunes by winning the
Champions League again.

Ten Hag will take the blame of course for the early exit and plenty of mistakes have been made, not least in the players the manager has been allowed to sign.
In reality assessing the job any United manager does with this team involves applying a series of filters. To be United manager is to undertake three concurrent
roles.

First you must manage the past, which is constantly in the room and which skews every act, achievement and expectation. Second you must manage the dysfunction
of the present, which expresses itself in every detail from a leaking stadium roof to the deep ills of a playing squad peopled with squatters and long‑term
missteps.
</item 43>

<item 44>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
The specifics of how that will work without Hayes’ involvement remain a mystery. It would be understandable for players to feel like they are still auditioning
for an absent director until May rolls around, while still knowing they must perform at the USWNT standard.

Along these same lines, there is at least the suggestion that the federation could be willing to sacrifice coherent preparation for this upcoming Olympic
tournament to focus more on the longer-term project of the 2027 World Cup. That itself represents a marked change from the expectations and pressure of constant
performance and winning that the team is known for.

Is that a good or a bad thing? Perhaps a little bit of both. The USWNT shouldn’t be ruled out of contention for the Olympics by any stretch, but this past
summer’s World Cup did reveal that the problems facing the team are far more foundational than just poor coaching decisions or the strange midfield chemistry.
Balancing realism with the pressure to win feels like a much more sustainable path forward for the USWNT.
</item 44>

<item 45>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Certain things [here] are different,” shrugs Ten Hag when asked why the No 1 is taking time to settle in England. “OK it is 11-v-11, and it is about attacking,
defending, transition. But there is more: the approach to the game, the speed of the game, the intensity. The varieties in styles. So there is absolutely a
difference and players need to adjust, need time to do so.”

Last season’s derby at Old Trafford ended in a 2-1 win for United. Bruno Fernandes’s equaliser was controversial with Marcus Rashford offside but not so the
goal was chalked off as he had not touched the ball. So far this term the captain has scored two goals in 12 United appearances and Ten Hag has no doubt he can
be a major factor in Sunday’s showdown.

“If he does [score] again, I’ll be happy,” the manager says. “Bruno is an [effective] player when we have a lot of the ball.
</item 45>

<item 46>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 46>

<item 47>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 47>

<item 48>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Mauricio Pochettino said he made Reece James Chelsea captain because of ‘his personality, his leadership, the respect he has from the club’. Photograph: Darren
Walsh/Chelsea FC/Getty Images

There is a reason why Pochettino calls Arteta “family”. They joined PSG at the same time, early 2001; Pochettino was 28, Arteta 18. They hit it off immediately,
Pochettino wowed by Arteta’s charisma and his reading of the game. “Already, he was a coach,” Pochettino said, and he has no qualms with what some perceive as
Arteta’s histrionic touchline behaviour.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 48>

<item 49>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
The manager’s future

Erik ten Hag’s team are treading water at precisely the wrong moment: as Sir Jim Ratcliffe takes charge of Manchester United’s sporting department before making
sweeping changes in the posts that will directly affect the manager. Richard Arnold has left despite officially staying until the end of the year and into his
role may come Jean-Claude Blanc, the CEO of Ratcliffe’s Ineos Sport, as the man Ten Hag reports to. Ten Hag’s line manager may change too, as Ratcliffe
considers whether a new football director is required, a position held by John Murtough. Ten Hag may have a first clue of his status in Ratcliffe’s new empire
if he is sounded out for his view of a prime factor in his ability to coach and manage the side to success: United’s transfer policy. This leads us to Murtough
…

Will Murtough survive?

The football director has no plans to depart but whether he remains in situ may be a matter for Ratcliffe.
</item 49>

<item 50>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"I had to set some standards"



Erik ten Hag claims there was not a good culture among staff and players before he joined Man United 😬pic.twitter.com/DJWe42mDXY — Sky Sports Premier League
(@SkySportsPL) December 5, 2023

Man United vs Chelsea lineups

Midfielder Casemiro and defender Lisandro Martinez remain out for Manchester United, while Christian Eriksen may not return in 2023.

Mason Mount missed United's defeat at Newcastle United on Saturday and Jonny Evans has not played for almost a month because of a thigh issue. Amad Diallo and
Tyrell Malacia have long-term injuries.

Man United lineup (4-2-3-1, right to left): Onana (GK) — Dalot, Maguire, Lindelof, Shaw — Amrabat, McTominay — Antony, Bruno Fernandes, Garnacho — Hojlund.
</item 50>
=== END ITEM id=a1cd67c51ac320e2 ===

=== ITEM id=769a8a6ca8ce1725 ===
Gold sentences and facts:
Sentence 0: Halloween isn’t the only holiday on the horizon now that October is drawing to a close.
  Fact 0.0: Halloween is not the only holiday on the horizon.
  Fact 0.1: October is drawing to a close.
Sentence 1: If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
  Fact 1.0: The reader may be in a new relationship this holiday season.
  Fact 1.1: The reader may be faced with the opportunity to introduce their significant other to their family this holiday season.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
What’s particularly great about this strip, though, is that it supports the newer Matter protocol, which allows you to control the lights with a wide array of
smart home platforms.

Govee LED Strip Light M1 Matter $ 45 $ 60 25 % off $ 45 $ 45 $ 60 25 % off The customizable Govee M1 Matter Compatible LED Strip Light is 6.56 feet long and can
display millions of colors and shades of white. You can also use it to create light patterns that sync with music. $45 at Amazon$48 at Govee

Though they launched last year, Google’s flagship wireless earbuds continue to impress us with their terrific sound quality and good noise cancellation, which
is why they’re one of our favorite pairs of wireless earbuds. We’re also fans because of all the other perks they offer, like multipoint connectivity, long
battery life, and a good transparency mode.
</item 2>

<item 3>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
However, it’s likely you’ve found yourself mulling over an array of reasons why the holidays may not be the right time for such an introduction.

The holiday season is often perceived as the most wonderful time of year, but that’s not necessarily the case for everyone.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
So when fellow video extraordinaire Vjeran Pavic (The Verge’s supervising producer) recently surprised me with a very cute birthday note and this magical little
gadget, I was elated.

And if that wasn’t enough, it has a carabiner hook. It is everything I have ever wanted in a rubber case and more. To have both my SD card reader and all my
cards in one place is priceless (well, actually $39.95). Thank you, Vjeran. <3

A mobile tripod for holiday snaps

Jess Weatherbed, news writer

When you go on holiday with your partner or family, you generally get stuck with three options when it comes to taking group photographs: a cramped selfie,
leaving someone out to take the picture, or asking a total stranger to take it for you. Not to be dramatic or anything, but I’d rather not ruin my vacation by
trying to figure out which individuals nearby are the least likely to dip the minute I hand over my phone.
</item 6>

<item 7>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
However, if you want to keep track of all the ghosties and ghoulies that’ll soon pay your front door a visit on Halloween, you can pick up the Blink Video
Doorbell with the latest Blink Outdoor security camera on Amazon for just $89.99 ($90 off).

Both of the 1080p cameras come with tech that’ll allow you to clearly see trick-or-treaters even after the sun goes down, in addition to support for motion
detection and two-way audio. The battery-powered Blink Outdoor 4 also covers a wider field of view and supports person detection, which wasn’t the case with the
prior model.

As budget picks, they lack some premium features, including smart alerts. You’ll also have to pay extra for features like cloud-based storage, though,
otherwise, they’re a pair of great security devices that’ll protect your home just fine for years to come.
</item 7>

<item 8>
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
</item 8>

<item 9>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Nevertheless, if you’re comfortable with these omissions, the first-gen model is still a good wearable to buy at this price.

Google Pixel Watch $ 200 $ 350 43 % off $ 200 $ 200 $ 350 43 % off Google’s first in-house smartwatch has a beautiful domed display and native Fitbit
integration for health tracking. The 41mm wearable comes with six months of Fitbit Premium and three months of YouTube Music. Verge readers can buy the wearable
at Wellbots for $199.99 when they use coupon code VERGEPIX80 at checkout. $200 at Wellbots

In case you missed it, you can still buy a couple of Blink security cameras and video doorbells for about half the price, which translates to some pretty good
deals on Blink Mini cameras and other gadgets.
</item 9>

<item 10>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Love Letter is on sale at Amazon for $6.29 (around $9 off). The fairly simple card game for two to six players is easy to get into and acts as a great intro to
more elaborate tabletop games.

Govee 100ft Smart LED Strip Lights $ 15 $ 25 40 % off $ 15 $ 15 $ 25 40 % off Govee’s basic 100ft LED smart light strip comes with 64 preset scenes so you can
create different vibes based on your emotions, holidays, nature, and more. It also offers various music modes and connects to your phone via Bluetooth or the
Govee Home app. $15 at Amazon

Tekton Everybit Tech Rescue Kit $ 22 $ 29 24 % off $ 22 $ 22 $ 29 24 % off A 46-piece screwdriver kit that can help you repair practically any tech device out
there.
</item 10>

<item 11>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 11>

<item 12>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Every Black Friday and Cyber Monday, there’s always a ton of hype about the discounts on big-ticket items — you know, the deals on the OLED TVs, game consoles,
and tablets of the world. But it never fails that many of the ultra-cheap items make for some of the most clutch deals, namely because spending very little on a
small accessory you end up using all the time can be the most gratifying.

This year, throughout the annual shopping weekend, we’re once again keeping an eye out for the best tech deals on those little gadgets and gizmos under $25.
We’ve also got roundups for those working with $50 or $100 budgets, but the items below are pretty much as cheap as you can get in the tech world without simply
buying absolute crap that’s destined to end up as e-waste.

Some of these helpful budget deals include gems like fully compatible MagSafe chargers for the latest iPhones, a colorful smart speaker, Apple’s handy AirTag
trackers, and some staff favorites.
</item 12>

<item 13>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Halloween isn’t the only holiday on the horizon now that October is drawing to a close. Once the clock strikes midnight on October 31st, Mariah Carey will
defrost, peppermint mochas will inevitably return, and the 2023 holiday season will have officially begun.

Thankfully for those looking to get in the festive spirit, Govee’s Matter-ready LED Strip Light M1 is on sale for an all-time low of $44.99 ($15 off) at Amazon
when you clip the on-page coupon. You can also buy it for $3 more direct from Govee through October 31st, which is still one of its better prices to date.

While handy throughout the year, Govee’s 6.56-foot (2m) light strip is particularly convenient during Halloween and other holidays. Not only can the bright,
tunable LEDs flash and blink to the beat of your music, but they can also act as a festive piece of decor that adds a pop of color to your holiday parties.
</item 13>

<item 14>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
Blink Video Doorbell (with Blink Outdoor 4) $ 90 $ 180 50 % off $ 90 $ 90 $ 180 50 % off Blink’s Video Doorbell is the best if you’re looking for a
budget-friendly buzzer that offers motion-activated recording and alerts, night vision, two-way audio, and up to two years of battery life. $90 at Amazon

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Some more deals to kickstart your weekend
</item 14>

<item 15>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
They deliver even more value to Pixel phone owners as well, who can enjoy exclusive access features like head tracking spatial audio and Google’s Clear Calling,
which reduces background noise during calls.

Google’s Pixel Buds Pro $ 117 $ 200 42 % off $ 117 $ 117 $ 200 42 % off Google’s Pixel Buds Pro are the company’s first earbuds to include active noise
cancellation. They combine impressive sound, great battery life, and good comfort — all without the connection issues of earlier models. And right now, Verge
readers can pick them up at Wellbots for $116.99 ($83 off) with offer code VERGE83. $117 at Wellbots

Speaking of Google gadgets, right now Verge readers can save an extra $80 on the already discounted Google Pixel Watch when they use promo code VERGEPIX80 at
Wellbots, which drops the checkout price to $199.99.
</item 15>

<item 16>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 16>

<item 17>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
When it comes to introducing your partner to your family, there’s still some extra planning to be done ahead of time. If you’re travelling this month, you may
need to decide on sleeping arrangements - as opposed to residing in your parent’s house - in order to minimise family stress.

“It would be really helpful to have conversations beforehand, so it’s clear what you are agreeing or not agreeing to,” Lurie explained. “For example, if your
family isn’t comfortable with you bringing someone home and sleeping in the same bedroom, then you and your partner may stay elsewhere. But those would be
conversations that you all need to have together. It really depends on your relationship with your family and with your partner.”

It’s also important to be checking in with your partner while spending time with your family. These check-ins may include some quality time together, so that
any family drama that may arise isn’t hurting your relationship.

“I’d recommend pulling your partner aside and saying: ‘How are we doing?’
</item 17>

<item 18>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
That isn’t necessarily a cause for concern, as long as you inform your significant other about the interworkings of your family dynamics.

“I think that we tend to be our best relational self when we’ve really done the work around understanding all of our past relationships,” Zar explained. “You
can tell your partner: ‘I don’t have a close relationship with my brother, and here’s why. Here are the intentional choices that I’ve made around it.’ Or you
can say: ‘Here’s my chosen family. These are the people who are important to me.’ Your significant other will see that you’ve been introspective and have
takeaways around how you’ve navigated your family’s origin.”

As long as you and your partner have discussed your relationship expectations - both before and during the holidays – introducing them to your family may deepen
your relationship and could be a step in the right direction.
</item 18>

<item 19>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 19>

<item 20>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 20>
=== END ITEM id=769a8a6ca8ce1725 ===

=== ITEM id=f294368762182c4f ===
Gold sentences and facts:
Sentence 0: Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.
  Fact 0.0: The Fortnite maker announced on September 28 that it is laying off 16% of its workforce.
  Fact 0.1: The layoff amounts to 870 people.
Sentence 1: Because while Google spent most of its first day attempting to explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.
  Fact 1.0: Google spent most of its first day attempting to explain complicated ins and outs of business.
  Fact 1.1: Epic was able to paint a black-and-white picture of good and evil.
  Fact 1.2: In that picture Epic itself was the clear underdog.
  Fact 1.3: Epic was able to do this because Google spent most of its first day on those explanations.

Retrieved items:
<item 1>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So this trial has been four years in the making. I look back at a September 2019 email that was in discovery about a plan to draw Google into a legal battle
over antitrust. Can you describe what it felt like, after four years, to hear the jury find for Epic?

Well, it’s a great day for all developers to see that the Sherman Antitrust Act works in the new era of tech monopolies; we’ve not had a major antitrust verdict
against a tech company that meant change and benefits for everybody since the 1990s, with the US v. Microsoft. Back in the early days of the internet. So this
is an awesome thing and it’s much needed by the industry which is being strangled by a few gatekeepers imposing insane amounts of control and extracting huge
taxes, which not only raise prices for consumers but also make a lot of kinds of products just unviable.

I understand the significance, but... you were there in person.
</item 2>

<item 3>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Snap

Laid off fewer than 20 project managers November 8. CNBC reported that Snap’s vice president of engineering, ​​Nima Khajehnouri, will also be leaving.

Amazon

Started cutting workers in its music streaming division in Latin America, North America and Europe, the company confirmed November 8.

Google

Confirmed November 8 it eliminated “a small number of roles” in its consumer support staff, Google Users & Products.

Virgin Galactic

Said in a memo to staff November 8 it is cutting costs and reducing staff. The number of employees impacted is currently unknown.

Ubisoft Montreal

Confirmed November 7 it will cut 98 roles in business administration and IT. Ubisoft said 124 positions will be eliminated overall across Canada, including
reductions in its global IT team and its VFX studio, Hybride.

Nextdoor

Revealed in its Q3 earnings that it is reducing its workforce by 25%.

Beyond Meat

Announced cuts November 6 affecting around 19% of its non-production employees.
</item 3>

<item 4>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This wasn’t the only layoff round of the year. Unity lowered its headcount three times in 2023, affecting about 900 jobs. In its quarterly financial results in
November, Unity reported a yearly revenue increase of 69 percent and it told investors, “We continued to manage costs well.”

Sony cut 100 jobs at Bungie, a company it bought for $3.6 billion in 2022. According to developers that are still there, Sony executives are attempting to use
this upheaval to wrest more control of the studio from Bungie founders and leaders.

Epic Games fired roughly 830 people this year, or 16 percent of its staff. This included significant job cuts at Mediatonic, the studio behind Fall Guys that
Epic purchased in 2021.

Mediatonic

“For a while now, we've been spending way more money than we earn,” CEO Tim Sweeney wrote about the layoffs.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Original story continues:

Llamacorn what now? What is happening?

Before we get to the Llamacorn, a little background. Epic is of course the studio behind Fortnite, the extraordinarily popular free-to-play game. Fortnite makes
money by selling in-game items with its virtual currency V-Bucks. Players often buy V-Bucks the same place they play Fortnite. And until August 13th, 2020, if
the player used an Android or iOS device and installed the game through an official app store, that purchase triggered an in-app payment fee to Google or Apple.

Critics call such fees the “Google tax” or the “Apple tax,” and Epic definitely wasn’t a fan.

When Epic decided to take action against these respective “taxes,” it made August 13th, 2020 a very busy day for Apple, Google, Epic, and us here at The Verge.
First, Epic announced it was bypassing Apple and Google’s app store fees.
</item 5>

<item 6>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Bandcamp

Eliminated 50% of its staff on October 16 after ownership changed from video game company Epic to music licensing platform Songtradr.

LinkedIn

Confirmed October 16 that it will cut 668 more jobs, bringing the total to nearly 1,400 this year, with the bulk of the most recent cuts impacting those with
R&D roles.

Flexport

On October 13 began laying off 20% of its workers, about 600 people, as it prepares for restructuring, and continues a cost-cutting campaign that began with
rescinding offers and working to lease office space.

Qualcomm

Is cutting 1,258 jobs in two of its California offices. According to a filing with the California Employment Development Department, the semiconductor company
plans to make the reductions in December.

Blue Origin

Laid off 40 people October 10, including software engineers and program managers, according to The Information.

Stitch Fix

Will lay off 558 employees after the personal styling subscription service closes a distribution center in Dallas.
</item 6>

<item 7>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 7>

<item 8>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Insightec

Is reportedly laying off 100 employees, with 60 from the company’s Israel headquarters.

Udaan

Reportedly laid off up to 150 employees in a restructuring effort. This month, the business-to-business e-commerce startup scored $340 million in new funding.

Enphase

Is reducing their workforce by around 10%, impacting roughly 350 contractors and employees, president and CEO Badri Kothandaraman wrote in a blog post December
18.

Arm Holdings PLC

Reportedly laid off more than 70 software engineers in China and will relocate some roles.

Delivery Hero

Is shutting down operations in Taiwan and Turkey and plans to eliminate roles at its Berlin headquarters, the company said December 18.

eBay

Is reportedly eliminating nearly 10% of its workforce in Israel. It’s the company’s second round of layoffs this year.

Intel

Will cut 235 employees at its Folsom office campus at the end of the year, the company reported to state employment officials December 15.
</item 8>

<item 9>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
100 Thieves

Is reducing its workforce by 20%, CEO John Robinson announced November 2. The esports brand is spinning out its game studio as well as its energy drink brand,
Juvee.

OpenSpace

Conducted a round of layoffs, CEO Jeevan Kalanithi announced November 2. The number of employees impacted by the layoffs has yet to be reported.

Informatica

Is laying off 10% of its global workforce, the cloud data management company announced in its Q3 results November 1.

Splunk

Is set to cut around 500 employees CEO Gary Steele said in a message to employees November 1 ahead of its planned acquisition by Cisco.

October 2023

Bungie

Is laying off about 100 employees, CEO Pete Parsons announced 10/30. The Sony-owned game studio is also delaying two of its upcoming titles as a result.

Karat

Reportedly cut an estimated 10% of its workforce October 27 after the financial startup for creators raised a $70 million Series B round earlier this year.
</item 9>

<item 10>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 10>

<item 11>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 11>

<item 12>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 12>

<item 13>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
CoinDesk

Announced on August 14 that it is cutting 16% of staff.

NCC Group

Confirmed on August 10 that it is making more layoffs, its second round of layoffs in just six months. The company declined to say how many employees are
affected by this round of cuts. NCC Group cut 125 employees in February.

Rapid7

Announced on August 9 that it plans to lay off 18% of its workforce, affecting more than 400 employees globally.

Mobile Premier League

Announced on August 8 that it is laying off approximately 50% of its employees or 350 people.

Astra

Announced on August 4 that it has laid off 25% of its workforce since the beginning of the quarter.

Discord

Announced on August 3 that it has let go of nearly 40 employees or 4% of its workforce.

HackerOne

Announced on August 2 that it is laying off 12% of its workforce, approximately 50 employees.
</item 13>

<item 14>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 14>

<item 15>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
He continued, “I had long been optimistic that we could power through this transition without layoffs, but in retrospect I see that this was unrealistic.”

Electronic Arts was one of the first video game companies to institute significant layoffs this year, with a reduction of 6 percent of its workforce, or about
800 employees, in March. EA later cut jobs at Dirt and F1 studio Codemasters, which it purchased in 2021 for $1.2 billion. EA culled an estimated 1,130 jobs in
2023.

CD Projekt RED and Sega each laid off about 100 people in the past 12 months, while Ubisoft fired an estimated 255 employees. Microsoft cut 10,000 jobs across
its businesses early in the year, and that included about 100 people at Halo studio 343 industries.

343 industries

These are just some of the biggest names in layoffs in 2023.
</item 15>

<item 16>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tropic

Eliminated 26 roles, CEO David Campbell confirmed in a LinkedIn post October 21.

Convoy

Is halting operations due to a “massive freight recession,” CEO Dan Lewis announced in a memo October 19, a week after laying off 20% of its workforce.

LegalZoom

Plans to lay off more than 100 workers in its sales division, the company disclosed in a Worker Adjustment and Retraining Notification Act letter.

Plume

Laid off more than two dozen workers, TechCrunch learned October 18, about a year after the trans healthcare startup raised a $24 million Series B.

Google

Eliminated at least 40 roles in its news division, Google confirmed October 18.

Expedia Group

Cut around 100 jobs, Skift reports, in its second round of layoffs in recent months. The cuts reportedly impacted employees across data and AI.

Stack Overflow

Laid off 28% of its staff, more than likely impacting 100 people, the Prosus-owned company announced October 16.
</item 16>

<item 17>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 17>

<item 18>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 18>

<item 19>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 19>

<item 20>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 20>

<item 21>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
No other teams are impacted.

Divvy Homes

Announced on September 11 that it is laying off 94 employees.

Sensor Tower

Announced on September 8 that it laid off approximately 40 people out of the 270+ at the company, including C-suite executives.

Roku

Announced September 6 that it plans to lay off approximately 10% of its workforce, representing over 300 employees.

August 2023

Malwarebytes

Announced on August 31 that it laid off 100 employees, mainly impacting corporate employees.

Twiga

Announced on August 21 that it is laying off 283 employees, or 33% of its workforce.

SecureWorks

Announced on August 14 that it will let go of 15% of its workforce, impacting about 300 employees. This is the company’s second round of layoffs this year.

CoinDesk

Announced on August 14 that it is cutting 16% of staff.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So there are absolutely no hard feelings, and we’re grateful that they joined the case because they did help in critical ways. Epic will continue to fight for
all developers, seeking remedies, and... I don’t know if Sundar is going to be calling me, but if he does, all of our discussions will be directed toward
solving the problem for everyone.

A lot has changed since you filed the original lawsuit over three years ago. For a bit there, it seemed like Epic had an interest in expanding outside of games
with acquisitions like Houseparty and Bandcamp. What changed? And did app store restrictions play into that at all?

Let’s see. No... well, not directly, anyway. We have a really broad strategy of building games and technology. We’ve had huge success in recent years with the
Unreal Engine gaining adoption among all kinds of industries even beyond games. We’ve also run into our own financial limitations. You know, we expanded the
company to nine times the size we were before Fortnite took off.
</item 22>

<item 23>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
September 2023

IronNet

In a regulatory filing published September 29, the cybersecurity startup has ceased all business activities as it prepares for Chapter 7 bankruptcy and has laid
off its remaining staff.

Naughty Dog

Is reportedly cutting at least 25 contract developers, two sources familiar with the situation told Kotaku.

Epic Games

Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.

Talkdesk

An impacted employee confirmed on September 27 that at least 140 people, if not more, were affected by Talkdesk’s third round of layoffs in less than 14 months.

Byju’s

Announced on September 26, the company plans to cut 5,000 jobs due to restructuring efforts. The startup has eliminated more than 10,000 positions in the past
two years.

Roblox

Confirmed to TechCrunch on September 21, Roblox let go of 30 employees in its talent acquisition organization. No other teams are impacted.
</item 23>

<item 24>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Multiverse

Will lay off nearly a third of their U.S. staff. It’s the second round of layoffs at the U.K.-based unicorn company in just over a month.

ONE

Is cutting its workforce by around 25% November 27, citing “market conditions” affecting the EV battery startup.

ByteDance

Started mass layoffs November 27 in its gaming department, Nuverse, after two years of tepid performance. The number of affected employees is unknown.

Veev

Is closing up shop, according to multiple reports, after reaching unicorn status in 2022.

Tulip

Laid off 25 employees, CEO Ali Asaria announced November 24.

Anar

Is shutting down operations and returning capital to investors, CEO Nishank Jain announced November 23.

Alerzo

Cut 100 workers November 21, reportedly due to increases in automation. The recent layoffs come after Alzero cut 15% of its workforce earlier this year and 5%
in 2022.
</item 24>

<item 25>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 25>

<item 26>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 26>

<item 27>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 27>

<item 28>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 28>

<item 29>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The company is now facing an unfair labor practice complaint, with workers alleging the layoffs are retaliation for unionizing.

Chewy

Laid off more than 200 employees November 14. The cuts impact roles in HR, recruiting, data, business intelligence, plus directors and higher managers.

Amazon

Is cutting “just over 180” roles in its gaming division, VP Christoph Hartmann wrote in an email to employees November 13.

Carta

Is reportedly cutting more of its workforce. The number of positions affected are unknown, but it would be the unicorn startup’s third layoff round this year.

Pico

Is cutting a “few hundred” roles, a person familiar with the matter told TechCrunch November 9. ByteDance’s Oculus challenger held an internal meeting
announcing it is halting its expansion this week.

Cruise

Started laying off contingent workers who supported driverless operations November 9. Those layoffs came after the company recalled 950 robotaxis and lost key
commercial permits to operate in California.
</item 29>

<item 30>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 30>
=== END ITEM id=f294368762182c4f ===
