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

=== ITEM id=d02d9e1dceda4d44 ===
Gold sentences and facts:
Sentence 0: McTominay is now United's top scorer for the season with six goals, one ahead of Bruno Fernandes and Rasmus Hojlund.
  Fact 0.0: McTominay is now United's top scorer for the season.
  Fact 0.1: McTominay has six goals for the season.
  Fact 0.2: McTominay is one goal ahead of Bruno Fernandes.
  Fact 0.3: McTominay is one goal ahead of Rasmus Hojlund.
Sentence 1: He has already been firing in many top-flight goals this term - as well as being the joint-leading scorer from the Champions League group stages.
  Fact 1.0: McTominay has already been firing in many top-flight goals this term.
  Fact 1.1: McTominay is also the joint-leading scorer from the Champions League group stages.

Retrieved items:
<item 1>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
In the final year of his contract at Parc des Princes, Real Madrid are favourites to secure a free transfer next summer, but they face competition from
Liverpool.

PSG have no games left to play in 2023 due to the French winter break, with Mbappe therefore unable to catch Ronaldo.

8 Mbappe has enjoyed a stunning 2023 having scored 52 goals for club and country Credit: AFP

2*. Harry Kane (Bayern Munich/Tottenham/England) - 52 goals

A stunning 2022/23 season saw Bayern splash out over £100m to sign the Spurs legend in August.

And Kane has repaid the Bundesliga club's faith having broken a number of records with his 25 goals so far this season.

The leading scorer [22 goals] in the German top-flight, the Three Lions skipper set a new Bundesliga record last month by finding the back of the net 17 times
in his opening 11 outings.
</item 1>

<item 2>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Previously he has never got more than seven. Where is this coming from? Partly from the fact that opponents have still not quite cottoned on to McTominay’s goal
threat, those late runs to the far post, the blur on the edge of the 18-yard box that you don’t spot until it’s too late. But partly, too, it is a function of
how Ten Hag is using him.

Erik ten Hag (right) and Mitchell van der Gaag, his assistant, talk tactics during the game. Photograph: Ash Donelon/Manchester United/Getty Images

Unlike his previous more conventional midfield roles, these days McTominay basically has a licence to create overloads wherever he finds them. Against low
blocks and structured defences – and nobody is remotely accusing Chelsea of possessing one of those – he provides the extra body in a tightly marked penalty
area.
</item 2>

<item 3>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Stream #PLGoalRush LIVE ???? https://t.co/BkLCmmvfda#OptusSport #PL pic.twitter.com/2wvzl1wjrG — Optus Sport (@OptusSport) November 12, 2023

Only Haaland – 13 – has scored more in the Premier League this season than the Egypt international, who took his tally to 10 by scoring for the sixth successive
home game to write another entry in club’s history books.

Advertisement

Diogo Jota’s late strike then put a quick end to questions over Liverpool’s mini-stumble after the draw at Luton and Europa League defeat in Toulouse.

Aston Villa moved to fifth, three points off the top, with their 3-1 win over Fulham at Villa Park, courtsey of an own goal from Antonee Robinson before strikes
from John McGinn and Ollie Watkins.
</item 3>

<item 4>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 4>

<item 5>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
We should do this on a consistent basis.

"We don't take the benefit in the attacking transition moments, we should be because the ability in the team can take benefit from it. We pay attention to it,
we need to grow. Then games will become easier."

So, what do the stats say?

Sky Sports' Sam Blitz:

Ten Hag is indeed right - Manchester United do rank very highly in their pressing stats in the final two-thirds of the pitch.

United are, as he says, "top of the Premier League" in terms of possessions won in the final third and high turnovers. He is slightly wrong in terms of winning
the ball in the middle third of the pitch - with United second behind Everton in that statistical table.

However, there is still work to be done in terms of improving United's pressing. Their passing permitted per defensive actions (PPDA) tally ranks 10th - so,
average - in the league.
</item 5>

<item 6>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Klopp said Tsimikas’ collarbone was “definitely broken”, adding he will be “out for a long time.”

Andros Townsend headed the winner for Luton Town as they beat Newcastle United in their first match since captain Tom Lockyer suffered a cardiac arrest agasint
Bournemouth a week earlier.

Advertisement

PURE ANGE-BALL! Richarlison scores against his old club to give Tottenham the lead! ????



"What a superb goal from Spurs – they just passed their way through Everton!" ????



Stream #PLGoalRush LIVE, or on-demand after, here ????
</item 6>

<item 7>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 7>

<item 8>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.
</item 8>

<item 9>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"I had to set some standards"



Erik ten Hag claims there was not a good culture among staff and players before he joined Man United 😬pic.twitter.com/DJWe42mDXY — Sky Sports Premier League
(@SkySportsPL) December 5, 2023

Man United vs Chelsea lineups

Midfielder Casemiro and defender Lisandro Martinez remain out for Manchester United, while Christian Eriksen may not return in 2023.

Mason Mount missed United's defeat at Newcastle United on Saturday and Jonny Evans has not played for almost a month because of a thigh issue. Amad Diallo and
Tyrell Malacia have long-term injuries.

Man United lineup (4-2-3-1, right to left): Onana (GK) — Dalot, Maguire, Lindelof, Shaw — Amrabat, McTominay — Antony, Bruno Fernandes, Garnacho — Hojlund.
</item 9>

<item 10>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Erik ten Hag smiles when asked if André Onana can be the “holding midfielder” for Manchester United described in Uefa’s technical report of the goalkeeper’s
display for Internazionale in May’s Champions League final.

“First the job of a keeper is to save, is to [guard] his goal, that’s his main job and don’t forget this,” says United’s manager. “But, yeah, he is a keeper who
we can use as an extra player and if we are developing [more] he can be more progressive and even higher up the pitch – an extra player, so that’s an
advantage.”

Onana’s last game for the Italian club was the 1-0 defeat by Manchester City in Istanbul as United’s crosstown rivals claimed a first Champions League and so
clinched the treble. Yet if he has recent experience of facing Pep Guardiola’s side, the 27-year-old will be a derby debutant against City in Sunday’s 191st
edition of the local squabble that will attract a global audience of millions.
</item 10>

<item 11>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
But Ten Hag says he has full faith that the England forward, who scored 30 goals last season, can find his scoring boots again.

"If you give him the trust and I play him every game, most games also he finishes the game. I have strong belief he will return to scoring a lot of goals," said
Ten Hag.

"It also has to do with the co-operation, with the movement around and the distribution to him. We have to work on that and that's my focus point."
</item 11>

<item 12>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Thiago Silva evened the score four minutes later with a header from Conor Gallagher’s corner.

Another former City player, Raheem Sterling, put Chelsea 2-1 up in the 37th by tapping in Reece James’ cross after a sweeping attack,

The score was even again in first-half stoppage time when Bernardo Silva crossed for Manuel Akanji to head in.

City re-took the lead two minutes after the break, with Haaland scoring again — sliding in to convert from close range.

Erling Haaland restores City's lead ???????? ???????????????????????????? after half-time! ⚡️



The Norwegian secures his 13th goal of the Premier League campaign in one of the games of the season so far! ????
</item 12>

<item 13>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
8 Kane already has 25 goals for Bayern in all competitions Credit: Getty

1. Cristiano Ronaldo (Al Nassr/Portugal) - 53 goals

Ronaldo has chipped in with 43 goals for Al Nassr to prove he never left his shooting boots back in Europe.

He has even banged in ten times for Portugal, as he helped the Euro 2016 winners triumph in all ten of their Euro 2024 qualifying matches.

His half-century in 2023 is the eighth time in his career that he has scored at least 50 goals in a calendar year.

His two strikes in the 5-2 win against Karim Benzema's Al Ittihad saw him leapfrog above Kane and Mbappe.

And with Al Nassr still having one match left before 2024, he could add to his ever-growing tally.
</item 13>

<item 14>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
The youngster then cut the ball back to Harry Maguire whose effort rebounded perfectly into the path of McTominay. The Scotland international took one touch
before blasting past Sanchez.

United led for the majority of the first 45 minutes but failed to build on their lead and were pegged back by the former Manchester City man, Cole Palmer. The
21-year-old, who made the transfer from the Etihad Stadium to Stamford Bridge in August, stroked the ball into the far corner with deadly accuracy. It was his
fifth Premier League goal of the season and his first from open play.

But United continued to put Chelsea under pressure in the second half and after numerous attempts, McTominay headed powerfully past Sanchez. The Scotsman had
seven shots, his most ever in a league game. Despite a flurry of late attacks driven by Reece James, Ten Hag's team stood strong and held on to see out the
victory.
</item 14>

<item 15>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Chelsea subs (12): To be confirmed

How to watch Man United vs Chelsea: TV channel, live stream

Here's how to watch this Premier League match across selected areas of the world's major regions:

Region TV Streaming USA — Peacock Canada — Fubo Canada UK — Amazon Prime Video Australia — Optus Sport India Star Sports 3 Asia Jio TV, Hotstar VIP

UK: This match will be available on Amazon Prime TV.

USA: This game is available for streaming via Peacock.

Canada: Every Premier League game this season is live streaming exclusively via Fubo in Canada, with a new FREE five-day trial for new subscribers.

Australia: Fans in Australia can stream every match live and on demand on Optus Sport.

MORE: Watch Premier League matches live with Fubo in USA
</item 15>

<item 16>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
We remember Onana also diving out of the way of the ball in Istanbul. Casemiro’s red card, lost leads in Manchester, Marcus Rashford unfairly convicted of a
venomous stamp in Copenhagen, when frankly, he’s just not that kind of semi-interested footballer.

And yet with both group games blank at half-time United were still just one goal away here from going through to the last 16, which really would have made for
an excellent punchline. It wasn’t to be. For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
Kimmich‑Goretzka‑Musiala is an A grade version of Amrabat‑McTominay‑Fernandes. Manuel Neuer is the thing Onana is trying so earnestly to be. Bayern actually did
go ahead and sign Harry Kane.

Harry Kane (centre) and his Bayern Munich teammates applaud fans after their Champions League victory at Old Trafford.
</item 16>

<item 17>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
There was not even the sense of heroic failure that accompanied their quarter-final exit to Bayern in 2010. There was just crushing failure. Because it wasn’t
Bayern who eliminated them. It was Galatasaray, who won at Old Trafford and scored six goals over 180 minutes against United. It was FC Copenhagen, who
qualified instead of them. Rasmus Hojlund’s former club will be in the last 16. His current one won’t. Copenhagen will be the anomalies in the draw, the
outsiders with the spot that had seemed reserved for United.

Manchester United’s defeat means they miss out on a spot in the Europa League (PA)

Instead, they are cast out of continental competition altogether. Europe’s sinking superpower have become just the fourth English team to prop up a Champions
League group when it concluded. There was something predictable about the manner of their demise; their defence was unlocked by Harry Kane, with a lovely flick.
</item 17>

<item 18>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Photograph: Adam Vaughan/EPA

They held the ball nicely, probed in neat, clipped, mannered attacks. A lot seemed to hinge on how desperately Bayern felt the need to twist that scalpel,
whether they could really be bothered to take this thing to the wall. United defended well. Their waves of pressing drew waves of applause. Everyone here seemed
to be trying.

But with 58 minutes gone Copenhagen scored and United’s hopes began to slip beneath the waves. Just over 10 minutes later Bayern made it 1-0 in Manchester.
Kingsley Coman got it, granted a huge expanse of time and space in front of the United goal, enough to set himself, as though taking a penalty, and belt the
ball into the corner. The moment was made by a delightfully sneaky round‑the‑corner pass from Kane. After all that pressure, all that tender hope, it looked a
bit like a training goal.
</item 18>

<item 19>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Goals from West Ham’s Jarrod Bowen and Mohammed Kudus extended Manchester United’s miserable run in the English Premier League.

Bowen bundled home at the second attempt in the 72nd minute, his 11th league goal this season, and Kudus struck his fourth goal in as many games to settle
matters.

It was another worrying setback for manager Erik ten Hag as United suffered a 13th defeat in 26 games in all competitions.

Facebook Twitter Whatsapp Reddit Email Share

A defensive howler from Manchester United and Mohammed Kudus capitalises! ????



What a finish from the West Ham man, but Kobbie Mainoo won't want to see this one back… ????



Stream LIVE, or on-demand after, here ????
</item 19>

<item 20>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Al Nassr striker Cristiano Ronaldo looks set to be the top scorer in 2023 after his Boxing Day double against Al Ittihad.

The 38-year-old may have swapped Europe for the Middle East, but he has showed no signs of slowing down in Saudi Arabia.

8 Ronaldo has been in lethal form for club and country in 2023 Credit: Getty

Ronaldo linked up with Al Nassr in the winter transfer window after parting ways with Manchester United following his explosive interview with Piers Morgan.

Liverpool legend Jamie Carragher described the five-time Ballon d'Or winner's switch to the Gulf nation as a 'sad end' to his career, but Ronaldo continues to
prove his doubters wrong.

But he isn't the only star that has been in great form, with players across Europe banging them in for fun.
</item 20>

<item 21>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Ten Hag on misfiring strikers: We need to be better

Please use Chrome browser for a more accessible video player Manchester United boss Erik ten Hag speaks on the improvement he's seen in goalkeeper Andre Onana
and insists he believes Marcus Rashford can get back to scoring goals

United go into Sunday's derby with a slight striker scoring problem. City forward Erling Haaland has scored more Premier League goals this season (9) than
United's five forwards have managed in all competitions (8).

United's top scorer this season is actually defensive midfielder Casemiro with four strikes so far.

Asked if their forwards are to blame for this frontline problem, Ten Hag replied: "Fault? It's co-operation. It has to click.

"We have shown in some games, we have so many in overload positions to the opponents goal and we don't net or don't even hit the target. We should be doing this
better."

It has been a sobering start to the season for Marcus Rashford, who has one goal so far this term for United.
</item 21>

<item 22>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
The performance this season was so bad that they find themselves out of Europe altogether as the bottom team in the group, although some at the club might find
solace from that. There wasn’t much else. The second place they aimed for has instead gone to Copenhagen, a club with a fraction of United’s budget. That is
where the damage was done.

While that meant there was only ever a slim hope of going through – and the Danish side’s victory over Galatasaray ensured any win would be irrelevant – the
club’s rich legacy in this competition was built on such defiance. There was absolutely none of it here.

There was nothing much at all. Where was the energy? Where was the intensity? The Stretford End eventually implored United to “attack attack attack” but the
complete absence of any such impulses eventually left some of the stadium booing the side off at the end.
</item 22>

<item 23>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
With two games left for City in 2023, Haaland has time to be the overall top scorer, though it remains to be seen if Pep Guardiola will play him against Everton
and Sheffield United.

Haaland has missed City's last five games with a foot injury, but it is believed he could be in contention now to return to action.

8 Haaland is the leading goalscorer in the Premier League this season

2*. Kylian Mbappe (Paris Saint-Germain/France) - 52 goals

Paris Saint Germain's all-time record scorer with 233 goals, it's easy to forget that the ex-Monaco man is still just 24 years old.

He has added even more goals to his tally for Les Parisiens this term, while he has fired in ten times for the French national team.

Mbappe's overall tally of 46 goals for Les Bleus is just ten shy of Giroud's current record.
</item 23>

<item 24>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 24>

<item 25>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
“Unfortunately, injuries tend to come when players are in good shape.”

In France, AS Monaco beat Marseille 3-2 to move top of Ligue 1, thanks to a brace from the midfielder Maghnes Akliouche and a powerful strike from Folarin
Balogun.

Marseille, who appointed Gennaro Gattuso as coach this week, took the lead after 30 secondsthrough Iliman Ndiaye’s goal, only for the hosts to respond minutes
later when Akliouche fired a low shot past Pau López.

The visitors regained the lead in the 18th minute when Ndiaye centred for Samuel Gigot to send a left-footed shot past Philipp Köhn, but Monaco equalised five
minutes later as Akliouche found Balogun who scored with a fierce finish.

Monaco sealed the win in the second half with another great strike from Akliouche.
</item 25>

<item 26>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Erik ten Hag has disagreed with Jamie Carragher's comments that Manchester United are going backwards this season - by saying the statistics show that his team
has actually improved.

United host local rivals Manchester City in a Super Sunday derby this weekend, live on Sky Sports with kick-off at 3.30pm, in the latest acid test to see how
far Ten Hag's side have improved from last season.

In his column in The Daily Telegraph, Carragher said United's summer signings have led to the club becoming "stale" - and that Ten Hag should now be questioned
about whether he can get the best out of his squad.

But in his pre-match press conference for the City game, Ten Hag hit back at those claims.

"In possession we have to do better, but there are reasons and I don't go into them, but everyone is seeing why. But there are also facts so I disagree [with
Carragher], we are much better," said Ten Hag.

"In high ball regains, we are a top of the Premier League.
</item 26>

<item 27>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
The England captain is the forward United neglected to bid for in the summer. The one they did buy, Hojlund, has at least struck in the Champions League but in
domestic competitions he has been outscored 18-0 by Kane.

Yet a rookie centre-forward is not the culprit in chief. For Ten Hag, this has been a chastening setback; with Sir Jim Ratcliffe set to finalise his investment,
with United listing in the Premier League, the chances are the Dutchman has managed them in the Champions League for the last time.

Go back four years and it required a 96th minute goal by Lucas Moura to stop him from reaching the final. Now the less heralded Lukas Lerager signalled United’s
demise, putting Copenhagen ahead against Galatasaray when United needed a draw to preserve their own chances of qualification; that, and a victory over Bayern.
</item 27>

<item 28>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Kane and Ronaldo, two national-team record scorers still operating in 2023 (Burak Akbulut/Anadolu Agency/Getty Images)

Considering how many hat-tricks Haaland scores for Manchester City, it’s not beyond the realms of possibility that he could score six goals during this
international break to move level with Juve, particularly given Norway’s first fixture is a home friendly against the Faroe Islands today (Thursday), before a
European Championship qualifier against Scotland in Glasgow on Sunday. That said, the Faroes’ defence is less leaky than you might expect — only twice in their
last 22 outings have they conceded more than three times in a game.

Advertisement

The wider question is whether we will ever see Haaland at a major tournament.

Despite the presence of him and Arsenal captain Martin Odegaard, Norway have failed to qualify from their Euro 2024 qualification group directly, with Spain and
Scotland already securing the top two spots.
</item 28>

<item 29>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

“After the break, we were more compact and certainly more energetic, hungrier and more determined,” Milan’s coach, Stefano Pioli, told DAZN. “That is a great
sign when playing every three days and it was a fully deserved victory.”

Milan suffered a potential injury blow as the midfielder Ruben Loftus-Cheek limped off in the first half. “It would be a shame if we lost Ruben, he felt a
twinge near the groin area,” Pioli said. “Unfortunately, injuries tend to come when players are in good shape.”
</item 29>

<item 30>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
It was already a drab game, with a muted crowd.

It didn’t help that Bruno Fernandes blazed United’s best opportunity over the bar after good work from Aaron Wan Bissaka. It looked like it was going to be one
of those games. It certainly wasn’t one of those big European nights at Old Trafford.

On 71 minutes, Bayern managed what they had been half-heartedly threatening all game. They also did it at near walking pace. Kingsley Coman had weaved his way
through the centre, for the ball to eventually be worked to Harry Kane. The English forward clipped an artful ball back through for Coman who finished
emphatically.

Kingsley Coman’s goal ensured Manchester United were knocked out of Europe (PA)

It was almost too easy for him. Why did Andre Onana stay so close to his line? If that sounds like the set-up for a joke… well, you can write your own
punchline.

This is the current reality for United.
</item 30>

<item 31>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Feyenoord 6 2 0 4 -1 6 4. Celtic 6 1 1 4 -10 4

Atletico Madrid and Lazio confirmed themselves as the two qualifiers for the Round of 16 before the final matchday. Top spot in Group E was decided by their
showdown in the Spanish capital on December 13, with Atleti winning 2-0. Feyenoord finish third and drop into the Europa League.

Results

Date Fixture/Result Sept. 19 Feyenoord 2-0 Celtic Sept. 19 Lazio 1-1 Atletico Oct. 4 Atletico 3-2 Feyenoord Oct. 4 Celtic 1-2 Lazio Oct. 25 Feyenoord 3-1 Lazio
Oct. 25 Celtic 2-2 Atletico Nov. 7 Atletico 6-0 Celtic Nov.
</item 31>

<item 32>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Santiago Gimenez (Feyenoord/Mexico) - 39 goals

One of the most exciting strikers in world football, the 22-year-old has taken European football by storm.

Joining Dutch side Feyenoord in the summer of 2022, he fired Arne Slot's men to the Eredivisie championship with 15 goals.

With 20 goals this season along with two strikes for Mexico, it's clear to see why Arsenal, Tottenham and Chelsea have all been linked with a move.

8 Gimenez seems destined for a big move in the near future Credit: Getty

5. Romelu Lukaku (Roma/Belgium) - 40 goals

After reuniting with Jose Mourinho at Roma in the summer, Lukaku has found his goalscoring form once again.

The Belgian has been firing back in Serie A following another loan move from Chelsea, netting in almost all of his starts this season.
</item 32>

<item 33>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Manchester United managers since Sir Alex Ferguson

If Manchester United were to appoint a new manager, that person would be the ninth coach to take charge of a United team following Sir Alex Ferguson's
retirement in May 2013, including caretaker/interim managers Ryan Giggs, Ralf Rangnick and Carrick.

Manager Honours Games David Moyes Community Shield 51 Ryan Giggs (caretaker) — 4 Louis van Gaal FA Cup 103 Jose Mourinho Community Shield, Europa League, League
Cup 144 Ole Gunnar Solskjaer — 168 Michael Carrick (caretaker) — 3 Ralf Rangnick (interim) — 29 Erik ten Hag League Cup *

*Still manager of the club
</item 33>

<item 34>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
28 PSG 1-1 Newcastle Nov. 28 Milan 1-3 Dortmund Dec. 13 Dortmund 1-1 PSG Dec. 13 Newcastle 1-2 Milan

Group G

Team GP W D L GD P 1. Man City (Q) 6 6 0 0 +11 18 2. RB Leipzig (Q) 6 4 0 2 +3 12 3. Young Boys 6 1 1 4 -6 4 4. Red Star Belgrade 6 0 1 5 -8 1

Manchester City and RB Leipzig confirmed their knockout places after four games played and City's 3-2 comeback win over the Bundesliga team on November 28
clinched top spot in Group G for Pep Guardiola. Young Boys drop into the Europa League.

Results

Date Fixture/Result Sept. 19 Young Boys 1-3 RB Leipzig Sept.
</item 34>

<item 35>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
United now have to beat Bayern Munich and hope Galatasaray draw with FC Copenhagen on the final matchday to progress, and it's the third time in the Champions
League this season that they have been in a winning position and then failed to emerge victorious.

Eight points have been dropped when they've been in the lead, and no other team has suffered a worse fate this term.

Captain Fernandes appears to be fed up of their inability to hold onto results as he told TNT Sports: "Every time we concede a goal we had many chances to score
again.

"I think at the end of the game we had like four chances to score a goal but we have to get our heads up and understand that sometimes it's about finding
another pass, finding the player that is free.

"Trying to get the best one in the best position and take the chance ourselves, we have to look for the team first and after, for ourselves."
</item 35>

<item 36>
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
</item 36>

<item 37>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"We started very good, proactive, brave, we created a lot of chances in that moment, we could have gone up with two or three goals."

Man United vs Chelsea final score

Final score Goalscorers Man United 2 McTominay 19', 70' Chelsea 1 Palmer 45'

Man United vs Chelsea live updates, highlights from Premier League

FT: United can breathe a sigh of relief and hold onto a crucial three points!

90th minute: Broja hits the post! That was so close to the equaliser for Chelsea. There are four minutes of added time.

80th minute: Garnacho so nearly makes it 3-1 to United but his delicately finish toys with the goalline.

74th minute: An insane recovery run and block by Reece James denies Garnacho from doubling United's lead. He was picked out superbly by Onana to kickstart a
rapid counter by the hosts.

70th minute: GOAL! 2-1 United!
</item 37>

<item 38>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Man United subs (12): To be confirmed

Ben Chilwell remains unavailable for Chelsea and Malo Gusto has been missing with a knee injury, but fellow defender Reece James is available again after
suspension, as is Marc Cucurella.

Romeo Lavia and Christopher Nkunku are likely to still be sidelined and Conor Gallagher is banned after receiving two yellow cards against Brighton. Carney
Chukwuemeka, Trevoh Chalobah and Wesley Fofana are out.

Chelsea lineup (4-3-3, right to left): Sanchez (GK) — Disasi, Silva, Colwill, Cucurella — Sterling, Fernandez, Caicedo — Mudryk, Palmer, Jackson.
</item 38>

<item 39>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.
</item 39>

<item 40>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
They are, at least, likely to qualify for the play-offs, and therefore will have two must-win games in March to secure their first major tournament appearance
since 2000 — the summer when Haaland was born. But there’s been little in recent performances to think Norway will breeze through those play-offs.

Juve’s individual record will soon be surpassed, but captaining his side to a bronze medal at the Olympics may stand as his nation’s greatest achievement for
much longer.

GO DEEPER Erling Haaland is phenomenal - so why hasn’t he made Manchester City better?

(Top photo: Sebastian Widmann – UEFA/UEFA via Getty Images)
</item 40>

<item 41>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Onana has been criticised for his role in Galatasaray's second goal as he allowed Ziyech's shot to spill under him and into the net, although Erik ten Hag says
collectively the defence isn't up to scratch.

"We're winning and then we're losing, we should have taken three points," the United boss told TNT Sports. "But there's also in more games, at home against
Galatasaray, away in Copenhagen, we play so good so I have to make a big compliment to the team.

"But also in the same moment I have to criticise the team and us as coaching staff, me as a manager. We aren't defending good enough, we are leading 3-1 and we
can't afford such mistakes because that makes the difference."

4 Onana had another disappointing performance in goal for United Credit: Getty

"Every goal has it's own story," he added.
</item 41>

<item 42>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 42>

<item 43>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Manchester United got themselves back to winning ways thanks to a superb Scott McTominay brace against Chelsea.

The victory remarkably puts them within three points of rivals Manchester City, after the reigning champions were defeated by Aston Villa, and keeps them in
reach of the top four places.

McTominay is now United's top scorer for the season with six goals, one ahead of Bruno Fernandes and Rasmus Hojlund. The Red Devils' captain had a first-half
penalty saved by Robert Sanchez and United's Danish striker was substituted after another Premier League game without a goal, despite making 11 appearances.

Erik ten Hag's side were dominant in the first half and finally pounced on Chelsea's inability to take control in midfield. Mauricio Pochettino's men attempted
to break at pace but Sofyan Amrabat made a vital interception and Fernandes found Alejandro Garnacho out wide.
</item 43>

<item 44>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
United fans will no doubt note the rich irony in the fact that these days McTominay is basically doing the job they once paid Paul Pogba to do. Marauding
midfield engine and late-running goal threat; the all-action superhero who can chip in with 10 goals a season and pull off slide tackles that make the highlight
reels. It never quite worked out for Pogba at United, largely – if you believe his most vehement critics – on account of his hair. McTominay is a different kind
of player with a different kind of skillset. But for all the undoubted leadership qualities of Bruno Fernandes and Harry Maguire, tactically it is McTominay
running the show right now: a development with numerous interesting consequences.

Here he scored his fifth and sixth goals of the season. Previously he has never got more than seven. Where is this coming from?
</item 44>

<item 45>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Harry Kane, for so long billed as a potential United player, didn’t score but did set it up.

This game showed one reason why he is at Germany’s biggest club rather than England’s. His side are through with ease, among the favourites for the competition
even if they do look lacking in a few areas.

Harry Kane’s assist was another reminder of what could have been for United (PA)

Not that you could judge from this game. It was almost a kickabout for them.

Bayern were still having more of the ball and the attacks. This was another tension for Ten Hag to overcome on the night. United were playing in quite a
controlled way when they were going to eventually have to go for it, but Bayern were still getting through that with relative ease.

It meant the team almost looked like they were in two minds, to go with how thoughts were always going to be in Copenhagen. The news came through shortly before
the hour that Lukas Lerager had scored for the Danish side, although it didn’t really change the atmosphere or the energy.
</item 45>

<item 46>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
That he can shout down the noise in the short term while building something in the long term. Right now, in the raptured afterglow of a big win, it feels like a
gamble being won. But best to come back and check in a few days, just to make sure.
</item 46>

<item 47>
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
</item 47>

<item 48>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
His latest acrobatic effort is well wide of the goal.

49th minute: Jackson has a glorious chance at the back post to make it 2-1 but he can't keep his header down!

46th minute: Both sides make a change at the break. Sergio Reguilon comes on for Lindelof and much to the delight of the Chelsea fans, Reece James comes on for
Cucurella.

HT: That's all from a hectic first half, hopefully more of it to come in the second!

45th minute: GOAL! 1-1! The former City man finds the bottom corner with a delicate stroke past Onana. It's a wonderful goal from Cole Palmer to pull level
right before the break.

36th minute: And that should be the equaliser! Mudryk finds himself in the box on his weak foot but the Chelsea forward drags his effort wide.

32nd minute: McTominay should have his second!
</item 48>

<item 49>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
70th minute: GOAL! 2-1 United! And it's McTominay who has a brace! It's a great ball to the back post by Garnacho and McTominay meets the header from inside the
six-yard box. He's in terrific goalscoring form and becomes United's top scorer this season.

69th minute: Antony tries one of his trademark curling efforts from the right after cutting in on his left but it's wide of the goal.

62nd minute: Penalty appeal for United! Shaw rockets an attempt goalwards and it looked as though Colwill may have blocked it with his hand. VAR took a quick
look and play continues.

61st minute: Antony's trivela is deflected for a corner, it may have been going wide.

54th minute: Unfortunately for Garnacho, lightning doesn't strike twice. His latest acrobatic effort is well wide of the goal.
</item 49>

<item 50>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Bayern Munich struck twice in the second half to come from two goals down and snatch a 2-2 draw at RB Leipzig but the champions dropped to third in the
Bundesliga with their second draw of the season.

Goals from Leroy Sané and Harry Kane cancelled out Leipzig’s first-half lead but the result left Bayern on 14 points, two behind the leaders, Bayer Leverkusen.
Leipzig, who host Manchester City in the Champions League next week, are fifth, a further point behind.

“Two very different halves from our perspective,” the Bayern coach, Thomas Tuchel, said. “In the first we started well but only for the first two minutes and
missed that huge chance.

“We were too slow, made too many mistakes and were not courageous enough. After the break we showed a different mentality. It was a bit wild, with a lot of risk
but it is a positive result, not to lose after being 2-0 down. We showed character.” Bayern travel to Copenhagen in the Champions League next week.
</item 50>

<item 51>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Varane was summoned from his inexplicable exile, after a 10-game absence when both Luke Shaw and Jonny Evans were preferred at centre-back to a World Cup
winner. He performed with an assurance to make his omission all the odder, but it has scarcely been the only U-turn in selection; with Harry Maguire and Scott
McTominay, Ten Hag is compiling a group who were first out of his plans and then back in. For a manager who likes to project an image of strong, decisiveness
leadership, he has had to reverse course rather too often. But then his team are going backwards.

When Coman scored, United became the first Premier League side to concede 15 goals in a Champions League group stage. When Maguire hobbled off, holding his
groin, and Evans came on, United had a man who played in United’s 2007-08 Champions League-winning campaign, dragooned into service 16 years later.
</item 51>

<item 52>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Ten Hag has just told MUTV that Chelsea do not "like to defend" and that United must take advantage of that this evening.

45 mins until kickoff: Raheem Sterling has failed to score in any of his 24 matches against Man United. Could tonight be the ex-Liverpool and Man City player's
night?

1 hour until kickoff: Team news is in! The eyebrow-raising decision from Erik ten Hag is that Marcus Rashford starts from the bench. Mauricio Pochettino starts
the former Man City man, Cole Palmer.

1 hour 30 mins until kickoff: Chelsea have won three of their last four Premier League away games but are without a league win at Old Trafford since 2013...

2 hours until kickoff: It's been a relatively busy start to the week off the pitch for Erik ten Hag, it's time to see what happens on it.

"I had to set some standards"
</item 52>

<item 53>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
It shows that while Ten Hag's side do win the ball high up the pitch, opponents do tend to keep the ball as well as play it around United's pressing forwards as
well.

What is PPDA? PPDA is the number of opposition passes allowed outside of the pressing team's own defensive third, divided by the number of defensive actions by
the pressing team outside of their own defensive third. A lower figure indicates a higher level of pressing, while a higher figure indicates a lower level of
pressing.

The other pressing concern United have, which Ten Hag did reference in Friday's press conference, is finishing chances when they win the ball high up.

United have the fourth-highest amounts of shots after winning the ball in their final third, but the glaring statistic is no goals from 104 high turnovers.

As expected, Manchester City have the highest amount of build-up attacks in the Premier League this season, so if United do win the ball high up in Sunday's
derby, they need to start taking those chances.
</item 53>

<item 54>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
With United out of Europe, there are questions about Erik ten Hag’s ability to coach the team (AFP via Getty Images)

These are not the sounds we are used to hearing after that Champions League theme. Something does stir when the anthem is played at Old Trafford amid the
lights. Well, for most, that is, except the players.

This is why there are growing questions for Erik ten Hag himself, even if everyone is constantly keen to put it in a wider context.

Manchester United should not be performing like this on what is supposed to be the classic European night at Old Trafford. This just reminded that all they are
going to have for some time are their memories.

It was another utterly forgettable display.

Bayern, already through but perhaps looking to make a point after their own 5-1 defeat to Eintracht Frankfurt, didn’t even need to step it up to win. They were
quite low-intensity themselves, Kingsley Coman’s winning goal coming from the forward almost walking through the middle of their team.
</item 54>

<item 55>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Norway’s Jostein Flo, a giant striker often utilised on the right flank, at the 1994 World Cup (Chris Cole/Allsport)

But perhaps the more pertinent thing about Norway is that, historically, they generally haven’t been very competitive.

They’ve only ever qualified for four major tournaments — in 1938, 1994, 1998 and 2000 — and have won a combined three matches in those appearances. They’re also
similar to Italy in that, at times, they’ve boasted various high-level strikers whose careers roughly overlapped — John Carew, Ole Gunnar Solskjaer, Tore Andre
Flo, Steffen Iversen — and at other points have suffered a complete lack of good centre-forwards.

At this point in time, Norway appear to have the most prolific striker in Europe, and it’s not unreasonable to consider where Haaland might end up in the
all-time international goalscoring charts worldwide, never mind just in relation to his compatriots.
</item 55>

<item 56>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Erling Haaland is just six away from Norway’s goalscoring record, so it is surely inevitable that he will soon be his nation’s most prolific striker ever.

On one hand, that will have happened remarkably quickly – Haaland only scored his first goals for Norway in 2020. On the other, this has been an incredibly long
time coming.

GO DEEPER What is Haaland really like?

Of the 213 FIFA-affiliated nations, Norway’s is the longest-standing outright individual goalscoring record in the world. It was set by Jorgen Juve, a
fascinating figure who ended his international career in 1937 and later became a renowned sports journalist in his home country.

Advertisement

Juve scored a relatively modest 33 goals in 45 matches for Norway, including five hat-tricks, although his tally is made remarkable by the fact he played as a
centre-forward in less than half of those games.
</item 56>

<item 57>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
Haaland is currently averaging nearly a goal a game for his country, which will inevitably be difficult for the 23-year-old to sustain over his career. But it’s
worth pointing out how impressive that is, even at this early stage. Again, excluding countries who have never qualified for a World Cup, only the
aforementioned quartet of Juve, Riva, Puskas and Nielsen, plus Japan’s Kunishige Kamamoto, hold their nation’s international goalscoring records and also boast
a rate of 0.75 goals per game or more.

Even Cristiano Ronaldo, the most prolific international goalscorer of all time with 127 for Portugal, boasts ‘only’ 0.63 goals per game, a lower rate than the
likes of Romelu Lukaku (Belgium), Kane and Aleksandar Mitrovic (Serbia), which owes to his early days as a winger rather than a central striker.
</item 57>

<item 58>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Photograph: David S.Bustamante/Soccrates Images/Getty Images

In Spain Real Madrid outclassed Girona 3-0 to return to the top of LaLiga, thanks to goals from Joselu, Aurélien Tchouaméni and Jude Bellingham.

Real moved on to 21 points from eight games, one ahead of Barcelona, who beat Sevilla 1-0 at home on Friday. Girona dropped to third.

“I feel very good. I have freedom of movement and I’m enjoying football. Whenever I put on this shirt I always try to do my best,” said Bellingham.

Real will also be buoyed up by the performance of Vinícius Jr, who looked lively in his first start since returning from a thigh muscle injury, before being
taken off in the 68th minute. Real’s coach, Carlo Ancelotti, said: “It was a solid, convincing game.
</item 58>

<item 59>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
4 Fernandes criticised their lack of ability to kill games off Credit: Getty

He added: "It's too bad to be honest, I don't want to be too negative but for what we've been doing in the Champions League is not enough, we have to control
our games every time.

"It's not the first game we've had like this, even with FC Copenhagen, I understand we had one player less but we scored to make it 3-2 and we had to secure the
result.

"We have to understand in these kind of games, in these kind of stages, you have to be smart enough and manage the game better."

United have also conceded 14 goals in the Champions League group stages, a new and unwanted record for the club that could get worse when they face Bayern in
the final match.
</item 59>

<item 60>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
The Portugal international Ronaldo has been battling with with Kylian Mbappe, Erling Haaland and Harry Kane for the title of the best goalscorer of 2023..

Here talkSPORT.com runs you through how Ronaldo compares to the top six highest scoring European based footballers this year...

7. Lautaro Martinez (Inter Milan) - 37 goals

All of Martinez's goals this year have come at club level for Inter Milan, with none coming on the international scene for Argentina.

With so many goals this campaign, it's no surprise that the Serie A giants are desperate to tie him down to a new deal.

He currently leads the Scudetto Golden Boot race - six more than closest rival Domenico Berardi.

8 Inter striker Martinez has carried on from last season by banging them in for fun Credit: Getty

6.
</item 60>

<item 61>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Another infamous European night at Old Trafford. It should be a landmark low, but almost the most deflating element of this 1-0 defeat to Bayern Munich was how
grimly routine it all felt.

Manchester United, one of the grandest and richest names in this competition, have again gone out of the Champions League at the group stage. It is the third
time it has happened since Sir Alex Ferguson, meaning it’s their most common round of exit in that period.
</item 61>

<item 62>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
They're easy saves and he's cost his team.

"If Onana doesn't make those two mistakes, I think United go on to win 4-1.

"He's so laid back on the ball that he can't switch on, he's late to dive for the free-kicks."

4 United now face a huge task in their hope of reaching the last 16 Credit: AFP

United's 12 goals in the group stage is the fifth highest tally out of any club but their defensive woes have been their downfall, and former defender Rio
Ferdinand cites a lack of balance as to why.

The 2007/08 Champions League winner with Man United said: "I think they've got to look at themselves.

"You can't go into a Champions League campaign and score nine goals in three away games and have one point at the end of it. I think that shows you that there's
a lack of balance in the team at the moment.
</item 62>

<item 63>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
“I think it was a decent 72 minutes where we didn’t take our chances. We should’ve gone in the lead but we didn’t and then it was one moment of switching off,”
he said.

“At this moment we are not top, but we have seen last week against the top of this league we were head-to-head. We didn’t score, we had the best opportunities
of the game.

“There we defended very well for 90 minutes, that has to be always our standard like what we did there. You have to bring that to every game on the pitch.”

Liverpool and Arsenal played out a thrilling 1-1 draw at Anfield that ensures the London side will top the English Premier League on Christmas Day.

Gabriel gives Arsenal a three and a half minute lead at Liverpool! ????



This top-of-the-table clash explodes into life early – and that's the Gunners' fastest ever goal at Anfield in #PL history! ⚡️
</item 63>

<item 64>
The Soccer Star on the Verge of Breaking an 89-Year-Old Record (The New York Times, 2023-11-17T21:47:19+00:00)
He was otherwise positioned in defence, from where he captained his nation to the bronze medal at the 1936 Olympics. That explains why his final international
goal came three years before his final cap, in June 1934. Therefore, it is likely that by the time Haaland scores six more goals, it will be around 90 years
since Juve reached the 33-goal mark.

There is technically one other record that stands for longer, also in Scandinavia.

Poul “Tist” Nielsen scored 52 goals in 38 games for Denmark between 1910 and 1925, although his record was equalled by Jon Dahl Tomasson — now manager of
Blackburn Rovers in the English Championship — in 2010. Tomasson elected to retire from international football after that year’s World Cup in South Africa
rather than seeking to make the record his own. Nielsen’s name therefore remains in the record books, although he now holds Denmark’s record only jointly.
</item 64>

<item 65>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Which, as the German champions have not lost in 40 group-stage games in this competition and United have not beaten anyone resembling a top team this season,
always felt unlikely.

This was far from their worst performance of the season, nowhere near as bad as Bournemouth or Crystal Palace or Newcastle, or the other defeat to Newcastle.
They were not as self-destructive as they had been in Denmark and Turkey and at home to Galatasaray. But they still lost.

Harry Maguire’s injury forced him off the pitch as 36-year-old Jonny Evans replaced him (EPA)

And they still went out, with four points, in fourth place. Go back a couple of months and Raphael Varane had said United could win the Champions League. On a
night when Ten Hag seemed to revise his opinion of the quadruple Champions League winner, the defender may need a rethink. Instead, United won a lone game in it
and even that required a penalty save to deny FC Copenhagen a draw.
</item 65>

<item 66>
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
</item 66>

<item 67>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
He has also found his shooting boots for his country, most notably scoring four in a 5-0 win over Azerbaijan in November.

Lukaku's career seems to have reignited and that has seen him close to claiming a place as one of the highest goalscorers this year.

8 Lukaku ended the Euro 2024 qualifying campaign as top scorer Credit: AFP

4. Erling Haaland (Manchester City/Norway) - 50 goals

No player in Premier League history has made more of an impact following their arrival than Haaland, who broke the goal record for a single season with 36
strikes last term.

He has already been firing in many top-flight goals this term - as well as being the joint-leading scorer from the Champions League group stages.

Just six out of his 50 goals in 2023 have been for his national side Norway, with Haaland helping the Citizens to a historic treble last campaign.
</item 67>

<item 68>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Names that don't make sense as next Man United manager

Julian Nagelsmann

While the German's name has come up in betting circles as a possibility for the next Manchester United manager, there's no sense to this actually taking place.
Nagelsmann was only recently hired as the head coach of the German national team, meaning there's no chance he is available until after Euro 2024 at the
earliest, and in that case, it's likely that Germany flopped at their own tournament, meaning his hire would probably be quite sour. This one just doesn't add
up.

Lionel Scaloni

While it's conceivable that the Argentine could become available at a moment's notice amid speculation of his potential exit with the defending World Cup
champions, this feels an unlikely pairing. Should he indeed leave Argentina, Scaloni has been open about his mental fatigue and is likely to take some time off
from management.
</item 68>

<item 69>
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
</item 69>

<item 70>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
7 Atletico 6-0 Celtic Nov. 7 Lazio 1-0 Feyenoord Nov. 28 Lazio 2-0 Celtic Nov. 28 Feyenoord 1-3 Atletico Dec. 13 Atletico 2-0 Lazio Dec. 13 Celtic 2-1 Feyenoord

Group F

Team GP W D L GD P 1. Bor. Dortmund (Q) 6 3 2 1 +3 11 2. PSG (Q) 6 2 2 2 +1 8 3. AC Milan 6 2 2 2 -3 8 4. Newcastle 6 1 2 3 -1 5

Group F confirmed one qualifier ahead of the final round of matches with the impressive Borussia Dortmund sealing their place in the Round of 16.
</item 70>

<item 71>
Top goalscorers of 2023: Cristiano Ronaldo brings up 50 goals with Erling Haaland and Harry Kane also high on list (TalkSport, 2023-12-12T22:36:24+00:00)
Kane, who has scored nine goals for England in 2023, is also the first player to register a hat-trick in their first outing against Borussia Dortmund.

It should come as no surprise that he is also on course to score the most goals in a debut Bundesliga campaign as he hunts down Uwe Seeler's record of 30 for
Hamburg in the 1963/64 season.

As well as that, Kane is on track to break Robert Lewandowski's record of 41 goals in a single season.

He's adding assists, too, as evidenced by his 'filthy' effort against Man United in the Champions League.

Like Mbappe, Kane cannot overtake Ronaldo's tally due to the German winter break.

8 Kane already has 25 goals for Bayern in all competitions Credit: Getty

1.
</item 71>

<item 72>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/wdM7tdgfb9#OptusSport pic.twitter.com/n7dqm1qnUH — Optus Sport (@OptusSport) December 23, 2023

Advertisement

Arsenal led through defender Gabriel’s fourth-minute header from Martin Odegaard’s free kick on Saturday (local time), but Mohamed Salah beat David Raya at his
near post to level in the 29th.

Liverpool came closest in the second half and hit the woodwork twice, through Trent Alexander-Arnold and substitute Harvey Elliott.

Mikel Arteta’s team go into Christmas a point clear of Liverpool and third-placed Aston Villa.

Liverpool defender Kostas Tsimikas broke his collarbone when he collided with Jurgen Klopp on the sideline.

Tsimikas was challenged by Arsenal forward Bukayo Saka and stumbled into Klopp, who fell on top of him.
</item 72>

<item 73>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Martínez then got his second goal after 77 minutes when he confidently finished a pass from Nicolò Barella with a first-time strike.

Martínez completed his hat-trick from the penalty spot with five minutes remaining after Matteo Lovato pulled Thuram’s shirt. He concluded the evening by
converting Carlos Augusto’s cross from 10 yards out.

Inter lead Serie A with 18 points, the same as their neighbours, Milan, who had briefly topped the table earlier in the day after beating Lazio 2-0 thanks to
second-half goals from Christian Pulisic and Noah Okafor. Rafael Leão played a vital role in both goals, with the 24-year-old Portugal international providing
two assists in a menacing display.

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.
</item 73>

<item 74>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 74>

<item 75>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
29 Galatasaray 3-3 Manchester United Nov. 29 Bayern Munich 0-0 FC Copenhagen Dec. 12 FC Copenhagen 1-0 Galatasaray Dec. 12 Manchester United 0-1 Bayern Munich

Group B

Team GP W D L GD P Arsenal (Q) 6 4 1 1 +12 13 PSV (Q) 6 2 3 1 -2 9 Lens 6 2 2 2 -5 8 Sevilla 6 0 2 4 -5 2

Arsenal finished top of the pile and their 6-0 thrashing of Lens on matchday five confirmed them as winners of Group B.

PSV's comeback win over Sevilla meant have also qualified thanks to Lens' defeat. The Ligue 1 side beat their LaLiga opponents 2-1 to make sure of a Europa
League berth as PSV and Arsenal played out an entertaining 1-1 draw.
</item 75>

<item 76>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 76>

<item 77>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
United have now won their past 19 Premier League games at Old Trafford when scoring first, including all 16 under their Dutch manager.

MORE: Every game on Amazon in December and how to watch

Ten Hag: Crisis? What crisis?

The United manager faced a demanding week amid reports indicating disunity within his team and his players did exceptionally well to block out the noise and
focus on footballing matters.

When asked about a potential crisis, he said: “Crisis? Not for us. We keep calm. We know where we have to build. We are going in the right direction. This team
is capable to be successful.”

And United were successful tonight, thanks to McTominay. Ten Hag explained how his midfielder has managed to become more involved in an attacking sense this
season: "He has a very good smell when to arrive and a very good finish.

"It is not a frustration, in the end, we won but we could have made our lives much easier," he said.
</item 77>

<item 78>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Chelsea have won the Champions League and Europa League twice each, while United have won the former trophy three times and the latter once.

United's tally of 12 FA Cup triumphs is eight more than Chelsea have claimed, but both sets of supporters will be hoping their teams can become superpowers
again after barren periods by their 21st-century standards.

The rise of mega-wealthy Manchester City to become arguably the best team in the world has not helped, and nor has Liverpool's resurgence under Jurgen Klopp.
United finished well off the pace albeit in a respectable third at the end of 2022/23, a season in which Chelsea came 12th – their lowest final position for 29
years.
</item 78>

<item 79>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
We showed character.” Bayern travel to Copenhagen in the Champions League next week.

Leipzig struck twice in six minutes with Loïs Openda putting them in front in the 20th when he completed a quick break. They scored again with Castello Lukeba
stabbing in after the goalkeeper Sven Ulreich misjudged a corner.

Bayern earned a 57th-minute penalty against the run of play for a handball and Kane cut the deficit with his eighth league goal. Only Manchester City’s Erling
Haaland, when he played for Borussia Dortmund, scored more often in his first six Bundesliga games with nine goals.

The Bavarians’ comeback was complete in the 70th with Kane clearing the ball in his own half and Musiala finding Sané for the equaliser.

Bellingham celebrates scoring Real Madrid’s third goal during their 3-0 away league win over Girona.
</item 79>

<item 80>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
"In high ball regains, we are a top of the Premier League. In the middle [of the pitch] ball regains, we are top. The pressing is very good. So that's not the
truth, what he [Carragher] is telling."

Please use Chrome browser for a more accessible video player FREE TO WATCH: Highlights from Sheffield United’s Premier League clash with Manchester United

In a fortnight where United have picked up three straight wins against Brentford, Sheffield United and FC Copenhagen, albeit via far from convincing
performances, Christian Eriksen said United still need to find their "balance and routines" this season.

Asked about those comments, Ten Hag said: "We are in the right direction. We have a way to go. I see positives but I see negatives.

"What Christian is referring to is we are not consistently in possession on the level where we can be. Like Arsenal away, the first 30 minutes in Bayern Munich
away. We should do this on a consistent basis.
</item 80>

<item 81>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
But Martínez opened the scoring in the 62nd minute when he collected Marcus Thuram’s low cross and chipped a finish over Guillermo Ochoa.

skip past newsletter promotion Sign up to Soccer with Jonathan Wilson Free weekly newsletter Jonathan Wilson brings expert analysis on the biggest stories from
European soccer Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

The substitute Lautaro Martínez converts a penalty against Salernitana. Photograph: Massimo Pica/EPA

Three minutes later Mateusz Legowski’s apparent equaliser was disallowed due to an offside.
</item 81>

<item 82>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.
</item 82>

<item 83>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Certain things [here] are different,” shrugs Ten Hag when asked why the No 1 is taking time to settle in England. “OK it is 11-v-11, and it is about attacking,
defending, transition. But there is more: the approach to the game, the speed of the game, the intensity. The varieties in styles. So there is absolutely a
difference and players need to adjust, need time to do so.”

Last season’s derby at Old Trafford ended in a 2-1 win for United. Bruno Fernandes’s equaliser was controversial with Marcus Rashford offside but not so the
goal was chalked off as he had not touched the ball. So far this term the captain has scored two goals in 12 United appearances and Ten Hag has no doubt he can
be a major factor in Sunday’s showdown.

“If he does [score] again, I’ll be happy,” the manager says. “Bruno is an [effective] player when we have a lot of the ball.
</item 83>

<item 84>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
9th minute: SAVED! It's a woeful penalty from Fernandes and Sanchez makes an easy diving save to his right.

7th minute: PENALTY! It's United who are rewarded for their bright start as Antony was brought down by Enzo Fernandez in the box. Chris Kavanagh originally
waved play on, however, VAR instructed him to take a look at the monitor and the referee gives the penalty.

4th minute: Sanchez makes a great save to deny Hojlund the opener, who was well found by Luke Shaw.

3rd minute: Dalot picks out Fernandes on the edge of the box who blasts well over the bar with a first-time strike.

1st minute: The visitors get us underway.

15 mins until kickoff: The teams have headed down the tunnel here at the Theatre of Dreams and kickoff is next...

30 mins until kickoff: The warm-ups are underway at Old Trafford on a chilly night.
</item 84>

<item 85>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."
</item 85>

<item 86>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Paris Saint-Germain join them in the knockouts after a 1-1 draw between the teams at Signal Iduna Park — a result that also confirmed Dortmund as group winners.

Newcastle were briefly in second spot during the final matchday before letting their lead slip to AC Milan, who snatched the Europa League spot in the process.

The Rossoneri finished level on points with PSG but behind them based on their head-to-head tiebreaker.

Results

Date Fixture/Result Sept. 19 Milan 0-0 Newcastle Sept. 19 PSG 2-0 Dortmund Oct. 4 Dortmund 0-0 Milan Oct. 4 Newcastle 4-1 PSG Oct. 25 PSG 3-0 Milan Oct. 25
Newcastle 0-1 Dortmund Nov. 7 Dortmund 2-0 Newcastle Nov. 7 Milan 2-1 PSG Nov. 28 PSG 1-1 Newcastle Nov. 28 Milan 1-3 Dortmund Dec.
</item 86>

<item 87>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream #PLGoalRush LIVE, or on-demand after, here ???? https://t.co/qE2Kd4Kh3N#OptusSport #PL pic.twitter.com/uRXhc07AAo — Optus Sport (@OptusSport) December
23, 2023

Richarlison scored for a third consecutive match against his old club Everton as Ange Postecoglou’s Tottenham ensured they would be in the EPL top four at
Christmas after a hard-fought 2-1 home win.

Sean Dyche’s visitors crucially had a Dominic Calvert-Lewin effort ruled out by VAR in the 51st minute which could have changed the complexion of the match.

Son Heung-min added to Richarlison’s opener and Andre Gomes scored a later consolation for Everton.

Burnley took all three points with a deserved 2-0 victory over hosts Fulham as Rebecca Welch became the first female referee to take charge of an EPL match.
</item 87>

<item 88>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 88>

<item 89>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Nostalgia underpinned Solskjaer’s management. Ten Hag cannot call upon his personal history in such a way; the Champions League final he almost reached was with
Ajax. His past does not offer the protection that means his name will always be sung at Old Trafford.

And, 24 years after Solskjaer’s stunning intervention, another United Champions League campaign finished against Bayern Munich. Not in glory, but ignominy. Like
Solskjaer, Kingsley Coman has a winner in a Champions League final; this time the winger finished off United, condemning them to a seventh home defeat of the
season. In a sense, it may have been the most respectable: United were vaguely competent on the night, but utterly incompetent over the course of a wretched
campaign.

There was no hint of a repeat of their most famous comeback against Bayern, ensuring Solskjaer’s name will forever be bracketed with Teddy Sheringham’s.
</item 89>

<item 90>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Wilson Odobert and Sander Berge scored to lift Burnley off the bottom of the table.

Dominic Solanke scored a hat-trick to earn Bournemouth a 3-2 win at Nottingham Forest and ruin Nuno Espirito Santo’s first game in charge at the City Ground.

Advertisement

Forest, who scored through Anthony Elanga and Chris Wood, had defender Willy Boly controversially sent off in the first half.

They were level until Solanke headed home in the fifth minute of stoppage time.

© AAP
</item 90>

<item 91>
VAR apology list in Premier League: Updated list of PGMOL apologies for referee mistakes in English football (Sporting News, 2023-09-30T21:27:02+00:00)
Referees once again took the headlines on the opening matchday of the 2023/24 Premier League season following an incorrect VAR decision in Manchester United's
fixture against Wolves.

Gary O'Neil's side were denied a late penalty after a foul from Andre Onana on Sasa Kalajdzic. Despite a long VAR check, Wolves were not given the spot-kick in
injury time and lost the match 1-0.

Fans watching the game were in disbelief when the call was made, and similar incidents have happened since, including against United. Bruno Fernandes said he
would wait for an apology — one that never arrived — after Cristian Romero was not punished for handball in the Red Devils' 2-0 loss to Tottenham, and plenty of
onlookers debated the penalty won by Marcus Rashford in United's subsequent comeback win at home to Nottingham Forest.
</item 91>

<item 92>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
This is the current reality for United. They obviously have many more issues than the goalkeeper – as illustrated by the goal itself – but a strange season for
Ten Hag’s primary signing only continues. United’s problems meanwhile combined. There were so many errors leading to the goal, to go with all of those that have
characterised this entire European campaign.

That can now be talked about definitively, since it has been decisively ended. It is a low but just part of a decade of disappointment in the competition that
has seen this club rise to its most emotional heights.

This game never threatened to do that. United never threatened at all. They’d been involved in so many ludicrously chaotic matches that the biggest of all ended
up being a non-event.
</item 92>

<item 93>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
We defended well, we knew they could cause damage with the ball, they are a team that plays good football, it was important to defend well and we did that.”

Lautaro Martínez came off the bench to score all four goals for Internazionale as they won 4-0 at Salernitana tto return to the top of Serie A.

The hosts kept Inter at bay for an hour but Martínez came on nine minutes after the break to bag four goals in 27 minutes.

“I scored four goals but the important thing is that Inter won,” the striker, who has scored nine goals in seven league games and is Serie A’s top scorer, told
Sky Sport Italia. “It was crucial. We lost a game at San Siro [to Sassuolo] that I don’t think we deserved but we conceded two goals from our errors, so we had
to win today.”

While Inter dominated the early stages, Salernitana found their footing and managed to control the tempo in a goalless first half.
</item 93>

<item 94>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Sheffield United clawed their way off the bottom of the table with their 1-1 draw at Brighton, while Tomas Soucek grabbed a late winner as West Ham came from
behind at the London Stadium to beat Nottingham Forest 3-2.
</item 94>

<item 95>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
The UEFA Champions League has reached the end of its group stage, with 16 precious places in the first knockout round now claimed.

Manchester City are looking to go back-to-back after their triumph in last season's Champions League, potentially becoming the first English club since
Nottingham Forest in 1980 to successfully defend the trophy.

The likes of Bayern Munich, Real Madrid and Barcelona have also enjoyed strong group-stage campaigns and will surely be in the mix towards the sharp end of the
competition. Atletico Madrid and Real Sociedad will complete the Spanish contingent in the next round.

Manchester United and Newcastle United will play no further part in Europe, though, having been eliminated by finishing bottom of their groups. Paris
Saint-Germain will be in the Round of 16, though.

The Sporting News reviews the group stage ahead of its final match night and examines what teams from each group need to qualify for the Round of 16.
</item 95>

<item 96>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 96>

<item 97>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Galatasaray knew that if they beat Copenhagen at Parken they would go through but Lukas Lerager scored the only goal in a 1-0 win for the hosts

Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's
side.

Results

Date Fixture/Result Sept. 20 Galatasaray 2-2 FC Copenhagen Sept. 20 Bayern Munich 4-3 Manchester United Oct. 3 Manchester United 2-3 Galatasaray Oct. 3 FC
Copenhagen 1-2 Bayern Munich Oct. 24 Galatasaray 1-3 Bayern Munich Oct. 24 Manchester United 1-0 FC Copenhagen Nov. 8 Bayern Munich 2-1 Galatasaray Nov. 8 FC
Copenhagen 4-3 Manchester United Nov. 29 Galatasaray 3-3 Manchester United Nov.
</item 97>

<item 98>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 98>

<item 99>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

They sang the name of Erik ten Hag’s predecessor. Not in a gesture of dissent, but as a reminder of a time when Manchester United were Europe’s best and an
evening without which a certain Norwegian would never have become their manager two decades later. After all, who put the ball in the Germans’ net? Ole Gunnar
Solskjaer. But that was a different millennium, a different United. Nostalgia underpinned Solskjaer’s management.
</item 99>

<item 100>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
32nd minute: McTominay should have his second! Fernandes picks him out beautifully from the left but the Scotsman heads directly at Sanchez. Either side of the
Chelsea goalkeeper and United would have a two-goal lead.

25th minute: Penalty appeal for Chelsea! Sterling drops to the floor after very minimal contact with Victor Lindelof but neither the referee nor VAR buy his
theatrics.

19th minute: GOAL! 1-0 United! Scott McTominay finishes off a scrappy period of play and United have a deserved lead.

18th minute: Silva's block denies Antony the opener. He's not scored at Old Trafford since his debut...

15th minute: It's all gone a bit wild here as neither side appears to have a midfield capable of dictating the play. Chelsea have just had two glorious chances
but Mudryk couldn't beat Onana and Fernandes made a brilliant recovery run to prevent the other attack.

9th minute: SAVED!
</item 100>
=== END ITEM id=d02d9e1dceda4d44 ===

=== ITEM id=85322f09c1cbbd4f ===
Gold sentences and facts:
Sentence 0: Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that infiltrated its territory.
  Fact 0.0: Benjamin Netanyahu is the Israeli Prime Minister.
  Fact 0.1: Benjamin Netanyahu said that Israel had been at war since this morning.
  Fact 0.2: Benjamin Netanyahu said that Israel's first objective was to clear out the hostile forces that infiltrated its territory.
Sentence 1: There, you can find stories like this in-depth breakdown of what to expect from both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their arguments and what the takeaways were from the opening arguments.
  Fact 1.0: A source offers an in-depth breakdown of what to expect from both sides of the SBF case.
  Fact 1.1: The breakdown covers where the prosecution and defense could gain ground or fall short in their arguments in the SBF case.
  Fact 1.2: The breakdown covers the takeaways from the opening arguments in the SBF case.

Retrieved items:
<item 1>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
However, there’s no wireless charging, the battery life (five or so hours per charge) is mediocre and the IPX2 sweat-resistance rating is too weak for the gym.
Amazon’s companion app is rather bloated too. But those shouldn’t be deal-breakers for $35. We gave the Echo Buds a review score of 77 in July.

Anker Soundcore Space A40

The Anker Soundcore Space A40 is one of the best values in wireless earphones at its normal price of $79, but now it’s down to $54 for Black Friday. That ties
the lowest price we’ve tracked. The Space A40 tops our guide to the best budget wireless earbuds because it delivers most of the features we expect from
headphones that cost twice as much. It’s comfortable, its ANC is genuinely capable, its battery lasts up to 10 hours per charge, its tiny case supports wireless
charging and it can pair with two devices at once.
</item 1>

<item 2>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Sony's WF-1000XM5 (7/10, WIRED Review) source multiple generations of great sound, top features, and excellent noise canceling, all wrapped into their most
compact presentation yet. They're not cheap, so this discount is key if you're looking to be an earlier adopter of Sony's latest flagship buds.

The only issues we had with Sennheiser's HD 660S2 (7/10, WIRED Review) bass-enriched audiophile beauties were their high price and stiff clamping force. At this
price, we can live with a tighter fit, especially since the payoff is gloriously clear and richly detailed sound.

Sony's working-class noise cancelers offer solid performance. clear sound, and impressive battery life at a much more reasonable price than the brand's top
offerings. With this sale, they offer some of the best value in their class from one of the top names in the business.
</item 2>

<item 3>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Like many Sony headphones, it also comes with a number of useful bonus features; “Speak-to-Chat,” for instance, automatically pauses your music when you start
speaking.

Sony WF-1000XM5

For an in-ear pair, the Sony WF-1000XM5 is also on sale for $248 at Amazon, B&H and others. (Yes, Sony’s headphone naming scheme is bad.) That matches the
lowest price we’ve seen and comes in $52 below Sony’s usual list price. This set tops our list of the best wireless earbuds and earned a score of 87 in our
review this past July. Like the over-ear XM5 above, it offers a host of useful features, a warm yet customizable sound, powerful ANC and a comfortably snug fit
that naturally isolates a good chunk of outside noise on its own. Battery life should sit around seven to eight hours with average use.
</item 3>

<item 4>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget Editor Jon Turi found this pair to deliver more low-end depth than most of its peers, however. It can also last up to 10 hours per charge.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 4>

<item 5>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
As Sony’s priciest wireless headphones, the WH-1000XM5 had a lot to live up to following the remarkably popular WH-1000XM4. Luckily they deliver the goods with
upgraded noise canceling, a posh new design, and a more refined sound signature that adds up to a great package – especially on sale.

Bose’s aptly named QuietComfort Ultra (9/10, WIRED Recommends) arrived in October with only one notable problem: they’re very pricey. Luckily they’ve already
gotten a discount, which makes it easier to land their class-leading noise canceling, great sound, and luxuriously comfy design that’s loaded with modern
features.

Photograph: Sony

Sony's WH-1000XM4 headphones (9/10, WIRED Recommends) have been a stalwart pair of wireless headphones for years now, and there's no secret to their success.
Offering excellent noise canceling, warm and detailed sound, and a ton of modern features in a comfy design, they're among the best travel headphones you can
buy.
</item 5>

<item 6>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is often a boon for deals on headphones and earbuds, and this year is no different. The shopping holiday has brought a plethora of deals on all
kinds of headphones, include several top picks from our guides to the best wireless headphones, wireless earbuds, budget wireless earbuds and gaming headsets.
Other popular models from the likes of Sony, Beats, Bose and Apple are discounted as well. All of this means it's an ideal time to pick up a new set, whether
it's to give as a gift to someone on your holiday shopping list or just to replace your own aging cans. To help you get a good bargain, we've rounded up the
best Black Friday deals on headphones and earbuds we could find for 2023.
</item 6>

<item 7>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
That's another $50 discount and an all-time low. We gave these in-ears a review score of 88 in September and recently named them the “best for noise
cancellation” pick in our wireless earbuds buying guide. As that title suggests, the main reason to consider this pair over the Sony WF-1000XM5 is its ANC
performance, which is the strongest we’ve tested from a true wireless model. We also enjoy Bose’s spatial audio tech. But the earpieces are on the larger side,
the call quality and battery life (at six-ish hours) are just OK and there’s no support for multi-device pairing.

Beats Fit Pro

The Beats Fit Pro is back down to $159, which isn’t an all-time low but is still roughly $20 below its average street price. This is the “best for workouts”
pick in our wireless earbuds buying guide, as it packs many of the iOS-friendly conveniences of Apple’s AirPods in a sportier and more stable design.
</item 7>

<item 8>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
If you prefer this kind of bulkier but easier-to-remove shape, this is an excellent value.

Anker Soundcore Space Q45

If you’re after a wallet-friendly pair of noise-canceling over-ear headphones, Anker’s Soundcore Space Q45 is worth a look at its Black Friday price of $100.
For that amount, this pair’s better-than-average ANC, softly padded design and multi-device pairing make for a strong value. Its V-shaped sound profile isn’t
especially detailed, but it can work well enough for pop music and it’s customizable through Anker’s app. Call quality is mediocre, however. This $50 discount
matches the lowest price we’ve tracked, and it's available at both Amazon and Anker's online store; just use the on-page coupon for the latter.

Audio-Technica ATH-M20xBT

The Audio-Technica ATH-M20xBT is a modest $10 off for Black Friday, bringing the wireless headphones down to $69.
</item 8>

<item 9>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Photograph: Technics

The Technics EAH-AZ80 (9/10, WIRED Recommends) are some of our favorite top-shelf earbuds thanks to their fantastic sound alongside great features and noise
canceling, and their ability to connect to up to three devices at once. This is one of the few sales we've seen all year, which makes their very high asking
price a lot more palatable.

Soundcore's Life Q30 are already impressive at full price thanks to noise canceling that punches well above its class, so this deal makes them something of a
steal. They add to the package with accessible and adjustable sound quality, good features, and great battery life.

Jump to a Topic: PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals

Smart Home Deals

Want to smarten up your home? See our guides to setting up your smart home, the best smart lighting, and the best mesh routers to get started.
</item 9>

<item 10>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Anker's Liberty 4 NC earbuds (8/10, WIRED Recommends) are another feature-packed pair of buds with great noise canceling, and are on sale for $80 ($20 off).

The Beats Studio Pro (7/10, WIRED Review) were released just a couple of months ago. We wish the battery life was longer and that the controls and EQ options
were more robust. But this price makes them more worthwhile, and they do have excellent noise canceling.

Photograph: Bose

These noise-cancelling headphones are our favorite for the office. With advanced signal processing and four microphones built in, these will limit any sound
around you, making them great for focusing on work in a busy environment and for Zoom calls. These often fluctuate in price, but we think it's a great deal.

We like Sony's standard LinkBuds (8/10, WIRED Recommends) for the open-ear design that allows you to hear what's going on around you.
</item 10>
=== END ITEM id=85322f09c1cbbd4f ===

=== ITEM id=18390d69942f96b7 ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley's sought-after voice on the promise and potential dangers of artificial intelligence.
  Fact 0.1: Altman had a sudden and mostly unexplained exit.
  Fact 0.2: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: This week's coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board's language.
  Fact 2.1: The prevailing theory is based on the way these giant tech companies work.

Retrieved items:

=== END ITEM id=18390d69942f96b7 ===

=== ITEM id=c21cb5fc238f37c8 ===
Gold sentences and facts:
Sentence 0: Even so, Kelce's 81.3 average receiving yards per game against the Raiders since 2019 suggests the All-Pro tight end is in line for a productive outing and could potentially snap his four-game scoring drought.
  Fact 0.0: Kelce averages 81.3 receiving yards per game against the Raiders since 2019.
  Fact 0.1: Kelce is an All-Pro tight end.
  Fact 0.2: Kelce's average receiving yards against the Raiders suggest he is in line for a productive outing.
  Fact 0.3: Kelce could potentially snap his four-game scoring drought.
Sentence 1: The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the classic game “Kiss, Marry, Kill”.
  Fact 1.0: The first link between Swift and Kelce was established in 2016.
  Fact 1.1: In 2016, Kelce played the classic game "Kiss, Marry, Kill" and that is when the first link between Swift and Kelce was established.
  Fact 1.2: Kelce helped bring his team to Super Bowl victory in 2020.
  Fact 1.3: Kelce helped bring his team to Super Bowl victory in 2023.

Retrieved items:
<item 1>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 1>

<item 2>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
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
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 6>

<item 7>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 10>
=== END ITEM id=c21cb5fc238f37c8 ===
