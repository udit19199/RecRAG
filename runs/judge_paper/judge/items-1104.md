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

=== ITEM id=a6fd427dba028e74 ===
Gold sentences and facts:
Sentence 0: It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.
  Fact 0.0: It could mean some sort of progress.
  Fact 0.1: Ratcliffe was born in Manchester.
  Fact 0.2: Ratcliffe has a vision to revolutionise the club.
  Fact 0.3: The progress would happen if Ratcliffe follows through with his vision.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United gave a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
Garnacho scored with an incredible overhead kick and Mainoo impressed on what was his full Premier League debut, which could earn him a start in Turkey and an
education in European football.

Around them are players who have succeeded on hostile nights. But if they thought Goodison Park was unwelcoming, they will be in for a surprise on Wednesday and
will need to find communal coping mechanisms. Fail to do so and they will just have to live and learn for next time.
</item 1>

<item 2>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
However, it's also possible he decides to stay on for the 2026 World Cup cycle, particularly if England have a good tournament in Germany. In that instance, he
would only be available for double duty on a temporary basis, and that would simply be too messy to outweigh the benefits. Brazil are currently in that boat,
and it's all gone wrong for Fernando Diniz despite his Copa Libertadores victory with Fluminense.

Mauricio Pochettino

In the event Todd Boehly proves to sport just as itchy a trigger finger as his successor Roman Abramovich at Chelsea, it's possible that Mauricio Pochettino
becomes available should Manchester United look to move on from Erik ten Hag. But would he really be a sellable hire, having just failed in London after bombing
at PSG and departing Tottenham in acrimonious manner? It seems unlikely that United would be interested in someone who demands significant club control and will
have just been fired from a struggling league rival.
</item 2>

<item 3>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?
</item 3>

<item 4>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
That raises doubts over the future of current sporting director John Murtough and chief executive Richard Arnold, although manager Erik ten Hag is believed to
be on a solid footing despite United’s poor start to the season.

Expect Sir Dave Brailsford, the mastermind behind Team GB’s cycling success at the London 2012 Olympics and director of sport for Ratcliffe’s company INEOS, to
have a prominent role in any reshuffle.

Ratcliffe also has ambitious plans to upgrade Old Trafford and turn it into a 90,000-seat stadium.

He wants the deal ratified in time for the January transfer window, however due to FFP concerns, there is no guarantees United will be able to spend big.

Despite the fact he turns 71 a day before the board meeting, it appears Ratcliffe’s plans centre around long-term stability and success rather than a short-term
turnaround in United’s fortunes.
</item 4>

<item 5>
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
</item 5>

<item 6>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Lopetegui is most known for his time in charge of the Spanish national team, where he succeeded Vincente del Bosque, but he also has big club experience at Real
Madrid.

Lopetegui has also managed in the Premier League, seemingly on an upward trajectory at Wolves before abruptly leaving just days before the current season amidst
tension with the board. And therein lies the issue around Lopetegui: he has left multiple positions in mysterious and awkward circumstances, with his Wolves
exit reminiscent of his departure from the Spanish national team, where he suddenly departed just days before the start of the 2018 World Cup as news of his
discussions with Real Madrid came to light. Between those two incidents, and his failed tenures at Madrid and Sevilla, would he be worth the risk at Old
Trafford?
</item 6>

<item 7>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
The UEFA Champions League has reached the end of its group stage, with 16 precious places in the first knockout round now claimed.

Manchester City are looking to go back-to-back after their triumph in last season's Champions League, potentially becoming the first English club since
Nottingham Forest in 1980 to successfully defend the trophy.

The likes of Bayern Munich, Real Madrid and Barcelona have also enjoyed strong group-stage campaigns and will surely be in the mix towards the sharp end of the
competition. Atletico Madrid and Real Sociedad will complete the Spanish contingent in the next round.

Manchester United and Newcastle United will play no further part in Europe, though, having been eliminated by finishing bottom of their groups. Paris
Saint-Germain will be in the Round of 16, though.

The Sporting News reviews the group stage ahead of its final match night and examines what teams from each group need to qualify for the Round of 16.
</item 7>

<item 8>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 8>

<item 9>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
In the second half United’s back five was André Onana, Aaron Wan-Bissaka, Jonny Evans, Raphaël Varane and Diogo Dalot, an entirely random collection of players
signed, at various times, by Ten Hag, José Mourinho, Ole Gunnar Solskjær and Alex Ferguson. Presumably Daley Thompson, Michael Portillo and Howard from Take
That were busy.

By the end United had five academy players on the pitch, ages ranged between 17 and 35, in outline evidence of internal good health, in reality another symptom
of a machine that wants to function properly but has for so long been made to judder along at half-speed. Ever feel like you’re starting to slip into the past
tense for good? When VAR happens at Old Trafford there isn’t even a VAR screen to say “VAR Check”. The 1990s were a hell of a decade. They still are around
here.
</item 9>

<item 10>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
However, names such as Alli, Heung-Min Son, Toby Alderweireld, Eric Dier and Kieran Trippier set the side up for a bright future, before chairman Daniel Levy
got in Mitchell’s way.

Leaving what he called his ‘dream job’ after less than two years, it was reportedly due to Levy’s over-involvement in negotiations and the recruitment process,
something which wasn’t an issue at his next club – Leipzig.

Ralf Rangnick landed Mitchell and more success followed. Signings such as Dani Olmo, Christopher Nkunku and Amadou Haidara again showcased what Mitchell can do
with the shackles off, and again the results followed.

6 Getting Nkunku from PSG was one of Mitchell's biggest coups Credit: Getty

Many of his signings starred in a run to the Champions League semi-finals in 2020, and in 2022 won the club’s first ever trophy with a DFB Pokal.
</item 10>

<item 11>
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
</item 11>

<item 12>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Photograph: Adam Vaughan/EPA

They held the ball nicely, probed in neat, clipped, mannered attacks. A lot seemed to hinge on how desperately Bayern felt the need to twist that scalpel,
whether they could really be bothered to take this thing to the wall. United defended well. Their waves of pressing drew waves of applause. Everyone here seemed
to be trying.

But with 58 minutes gone Copenhagen scored and United’s hopes began to slip beneath the waves. Just over 10 minutes later Bayern made it 1-0 in Manchester.
Kingsley Coman got it, granted a huge expanse of time and space in front of the United goal, enough to set himself, as though taking a penalty, and belt the
ball into the corner. The moment was made by a delightfully sneaky round‑the‑corner pass from Kane. After all that pressure, all that tender hope, it looked a
bit like a training goal.
</item 12>

<item 13>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
I think that shows you that there's a lack of balance in the team at the moment.

"Yes they can get goals and yes they are attacking at will and creating chances, many of them coming through Bruno. But I think defensively, they just made
mistakes, I think the goalkeeper today has to hold his hands up and say, 'Yep, I was at fault for a couple of goals.

"He should do better. But, I think the manager will be scratching his head and thinking, 'The game plan first-half worked perfectly,' and then all of a sudden
they score a goal, it changes the dynamic of the game.

"They were 2-0 up, 2-1, and you're thinking, 'Wow, I set the team up perfectly well, they execute it but then a mistake cost us,' and at 3-1, the same again.
</item 13>

<item 14>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Antonio Conte

Former Chelsea and Tottenham boss Conte hasn't cropped up near the top of any betting lists, but there could be some sense in hiring the Italian to take over at
Old Trafford. Conte has been out of work since his acrimonious departure from Tottenham this past summer, after he seemingly shoved his way out of the club
amidst reports he was utterly miserable in London.

While the idea of a control freak such as Conte taking over the Manchester United mess could easily be a recipe for disaster, the club could also seek out a
fixer of his nature, hoping for an immediate boost they can build on once things crumble under the Italian, as they often do. Still, the Red Devils would likely
have competition as he seems to prefer working in his home country.

Julen Lopetegui

Fifty-seven-year-old Spanish boss Lopetegui represents maybe the perfect mix of experience, availability, and interest as Manchester United seek out possible
new managers.
</item 14>

<item 15>
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
</item 15>

<item 16>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
TIMELINE OF THE TAKEOVER SAGA

It was in November 2022 as Qatar became the centre of the football universe during the controversial 2022 World Cup that the Glazers revealed they were looking
to cash in all or at least some of their shares.

The bombshell announcement was seen by the majority of Manchester United fans as the beginning of the end of the divisive and destructive regime that began with
a now-outlawed leveraged buyout in 2005 and saddled the club with debts that at one stage soared to as high as £1bn.

Brothers Joel and Avram, the two most hands-on of the six Glazer siblings who have overall control of United, had instructed the New York-based Raine banking
group to handle the bidding process.
</item 16>

<item 17>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Fulham are in a worse situation than the Red Devils. After a mixed set of results with 3 wins, 3 draws, and 4 defeats, they are sitting in 14th position in the
league table. However, they are still unbeaten in their last two games in all competitions. They won their last match by a 3-1 margin against Ipswich Town in
the EFL Cup round of 16. Before that, they drew 1-1 against Brighton in their last Premier League match. However, they will have a tough time holding onto the
streak in this match.

Manchester United Team News

Man United have been fighting a battle to line up their best XI as players fall victim to one injury after another. Casemiro, who got injured during the
international break, got back on the field against Newcastle United in the EFL Cup match. However, he was taken off at half-time with a new hamstring issue. He
will probably remain sidelined for weeks.
</item 17>

<item 18>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
In the event he does return quickly, he's been heavily linked to Real Madrid as a possible replacement for the Carlo Ancelotti, who is expected to become the
next permanent Brazil head coach.

It's hard to imagine that Scaloni, who seemingly craves stability and routine, has no experience at club level, and lacks ties to the Premier League or Old
Trafford (beyond a brief loan spell with West Ham as a player in 2006) would be a match for Manchester United in any capacity.

Gareth Southgate

You can sense a trend here as another currently employed national-team manager is mentioned in England boss Southgate. While he's been linked to the United job
before, it would take a perfect storm for him to be installed, whether on a permanent basis or as an interim stopgap option.

Some believe he could be considered next summer once the Euros are over: it has been speculated that Southgate could leave his England post following next
summer's tournament, with his contract expiring in December next year.
</item 18>

<item 19>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 19>

<item 20>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It was the sequel Newcastle United did not want, the second cruel comeback in as many Champions League games. For some 45 minutes, including half-time, they
could savour the sight of a table that showed them in second, permitting them to imagine a date with Real Madrid or Bayern Munich in the knockout stages. And
then, little more than half an hour, they were ejected from Europe altogether.

Christian Pulisic’s equaliser threatened to condemn them to the Europa League.
</item 20>

<item 21>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Already a top tier European owner of OGC Nice, there has been some success, but nothing compared to their French riviera neighbours, where Ratcliffe appears set
to look for inspiration.

Paul Mitchell recently left the principality of Monaco, but he could well be back there in meetings with Ratcliffe who is planning to appoint a sporting
director.

Mitchell fulfilled that role at Monaco after previous stints at Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s most respected
executives.

A former player in England’s lower leagues, Mitchell was previously linked with Chelsea and Liverpool, and much like Ratcliffe, he’s also from Greater
Manchester.

Starring mainly for Wigan and MK Dons, it was with the latter where Mitchell was moved upstairs, and made his first big move by spotting future England regular
Dele Alli after being named their head of recruitment in 2010.
</item 21>

<item 22>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
He had taken Miguel Almiron’s pass and teed himself up. His initial non-celebration may have reflected his own surprise at the unerring accuracy of his strike.

Theirs was the right response to the disappointment in Paris, a stirring effort. At this level, Newcastle do not possess a talent advantage but they can seek to
outwork anyone. They rattled Milan by running at them. It took two astonishing interventions to stop them scoring both before and after Joelinton’s strike.

At 0-0, Fikayo Tomori made a magnificent goal-line tackle to stop Miguel Almiron applying the finishing touch to Joelinton’s low cross. It was a remarkable
rescue act but pressure told when the Brazilian struck. Another almost joined him on the scoresheet at 1-1.

Newcastle saw their European dreams slip away (Getty)

Maignan tipped Bruno Guimaraes’ 20-yard shot onto the bar.
</item 22>

<item 23>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Fulham Team News

Fulham’s injury situation has not improved so far. Adama Traore is suffering from a thigh problem, Issa Diop is nursing a foot issue, Kenny Tete is out due to
an unspecified injury, and Tosin Adarabioyo is suffering from a groin injury.

However, Tete, Adarabioyo, and Traore are on their way to being fully fit. They could at least make the squad for this weekend’s high-stake match against Man
United. Yet, the coach’s biggest concern revolves around his No. 9 position. With his first goal in the EFL Cup match, Rodrigo Muniz proved that he can be a
good pick for the role on a regular basis.

The coach hinted that he could choose Muniz over Raul Jimenez who is yet to score since he joined Fulham this summer.
</item 23>

<item 24>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Ten Hag has just told MUTV that Chelsea do not "like to defend" and that United must take advantage of that this evening.

45 mins until kickoff: Raheem Sterling has failed to score in any of his 24 matches against Man United. Could tonight be the ex-Liverpool and Man City player's
night?

1 hour until kickoff: Team news is in! The eyebrow-raising decision from Erik ten Hag is that Marcus Rashford starts from the bench. Mauricio Pochettino starts
the former Man City man, Cole Palmer.

1 hour 30 mins until kickoff: Chelsea have won three of their last four Premier League away games but are without a league win at Old Trafford since 2013...

2 hours until kickoff: It's been a relatively busy start to the week off the pitch for Erik ten Hag, it's time to see what happens on it.

"I had to set some standards"
</item 24>

<item 25>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
In 2020‑21, they lost their final two group games and were eliminated, while back in 2015-16, Wolfsburg won the deciding group match to knock out United.

Mauro Icardi celebrates scoring the winner for Galatasaray in their 3-2 win at Old Trafford in early October. Photograph: Darren Staples/AFP/Getty Images

When it comes to pressure in the Champions League of late, United do not cope well and there are few events trickier than a night in Istanbul. Pallister smiled
at a bell boy at the team hotel and got a finger across the throat in response, Brian McClair had a cleaner bang on his bedroom ceiling in the night and other
players received numerous phone calls to disrupt their sleep. The current squad could face the same level of malevolence and need to be prepared.

United have done little learning in recent times, repeating the same failures in Europe’s elite competition.
</item 25>

<item 26>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
It is 30 years since Manchester United’s most famous trip to Istanbul with Champions League elimination on the line. Galatasaray created an unwelcome hell that
Cerberus would have been proud of and a night of sin ended with Alex Ferguson’s side being dumped out of Europe.

United, realistically, required a win to progress after drawing 3‑3 in the first leg of the second‑round tie at Old Trafford but the atmosphere in Turkey
subdued the visitors and the second leg stuttered to a goalless draw. Eric Cantona was sent off late on for his frustrations with the referee and the home
team’s addiction to timewasting. There was a post-match melee in the tunnel involving the United squad and riot police as the trip ended like it began – with
intimidation. Many of United’s team froze, unable to deal with the ferocious and vociferous atmosphere they faced.

The match on Wednesday is not a knockout game but in truth the scenario is not much different.
</item 26>

<item 27>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Chelsea have won the Champions League and Europa League twice each, while United have won the former trophy three times and the latter once.

United's tally of 12 FA Cup triumphs is eight more than Chelsea have claimed, but both sets of supporters will be hoping their teams can become superpowers
again after barren periods by their 21st-century standards.

The rise of mega-wealthy Manchester City to become arguably the best team in the world has not helped, and nor has Liverpool's resurgence under Jurgen Klopp.
United finished well off the pace albeit in a respectable third at the end of 2022/23, a season in which Chelsea came 12th – their lowest final position for 29
years.
</item 27>

<item 28>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"I had to set some standards"



Erik ten Hag claims there was not a good culture among staff and players before he joined Man United 😬pic.twitter.com/DJWe42mDXY — Sky Sports Premier League
(@SkySportsPL) December 5, 2023

Man United vs Chelsea lineups

Midfielder Casemiro and defender Lisandro Martinez remain out for Manchester United, while Christian Eriksen may not return in 2023.

Mason Mount missed United's defeat at Newcastle United on Saturday and Jonny Evans has not played for almost a month because of a thigh issue. Amad Diallo and
Tyrell Malacia have long-term injuries.

Man United lineup (4-2-3-1, right to left): Onana (GK) — Dalot, Maguire, Lindelof, Shaw — Amrabat, McTominay — Antony, Bruno Fernandes, Garnacho — Hojlund.
</item 28>

<item 29>
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
</item 29>

<item 30>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
28 PSG 1-1 Newcastle Nov. 28 Milan 1-3 Dortmund Dec. 13 Dortmund 1-1 PSG Dec. 13 Newcastle 1-2 Milan

Group G

Team GP W D L GD P 1. Man City (Q) 6 6 0 0 +11 18 2. RB Leipzig (Q) 6 4 0 2 +3 12 3. Young Boys 6 1 1 4 -6 4 4. Red Star Belgrade 6 0 1 5 -8 1

Manchester City and RB Leipzig confirmed their knockout places after four games played and City's 3-2 comeback win over the Bundesliga team on November 28
clinched top spot in Group G for Pep Guardiola. Young Boys drop into the Europa League.

Results

Date Fixture/Result Sept. 19 Young Boys 1-3 RB Leipzig Sept.
</item 30>

<item 31>
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
</item 31>

<item 32>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
The England captain is the forward United neglected to bid for in the summer. The one they did buy, Hojlund, has at least struck in the Champions League but in
domestic competitions he has been outscored 18-0 by Kane.

Yet a rookie centre-forward is not the culprit in chief. For Ten Hag, this has been a chastening setback; with Sir Jim Ratcliffe set to finalise his investment,
with United listing in the Premier League, the chances are the Dutchman has managed them in the Champions League for the last time.

Go back four years and it required a 96th minute goal by Lucas Moura to stop him from reaching the final. Now the less heralded Lukas Lerager signalled United’s
demise, putting Copenhagen ahead against Galatasaray when United needed a draw to preserve their own chances of qualification; that, and a victory over Bayern.
</item 32>

<item 33>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
There was not even the sense of heroic failure that accompanied their quarter-final exit to Bayern in 2010. There was just crushing failure. Because it wasn’t
Bayern who eliminated them. It was Galatasaray, who won at Old Trafford and scored six goals over 180 minutes against United. It was FC Copenhagen, who
qualified instead of them. Rasmus Hojlund’s former club will be in the last 16. His current one won’t. Copenhagen will be the anomalies in the draw, the
outsiders with the spot that had seemed reserved for United.

Manchester United’s defeat means they miss out on a spot in the Europa League (PA)

Instead, they are cast out of continental competition altogether. Europe’s sinking superpower have become just the fourth English team to prop up a Champions
League group when it concluded. There was something predictable about the manner of their demise; their defence was unlocked by Harry Kane, with a lovely flick.
</item 33>

<item 34>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 34>

<item 35>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Joel and Avram Glazer put the Red Devils up for sale in November 2022 Credit: AFP

5 United supporters are desperate to see the Glazers depart Old Trafford Credit: Twitter

The fact the Glazers will remain in situ has not gone down well with supporters, but this is seen as the first step to Ratcliffe eventually removing the
Americans and taking overall control.

The exact timescale as to when that happens remains vague and how much it will cost Ratcliffe overall could depend on United’s sporting and commercial success
in the coming seasons.

Thursday’s board meeting was pre-arranged and while it is expected that Ratcliffe’s proposal will discussed in detail, sources close to United have indicated to
talkSPORT that it is far from guaranteed it will be put to a vote with several details still to be thrashed out.
</item 35>

<item 36>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
The fears came true and they lost 3-0 in the Manchester derby. They had some hope in their next match in the EFL Cup round of 16 match against Newcastle United.

The boys in blue 🤩 pic.twitter.com/0awNdmkXtT — Manchester City (@ManCity) November 3, 2023

Especially since they defeated the Magpies last term in the final of the EFL Cup. However, the reigning Carabao Cup champions could not defend their title any
longer. They lost by another 3-0 margin to Newcastle United. With 5 wins and 5 defeats in the domestic league, they are now sitting in 8th position in the
table. It still seems that coach Erik Ten Hag has the backing of the management. However, he will still have to pull up his team from the doom as soon as
possible.

Fulham are in a worse situation than the Red Devils.
</item 36>

<item 37>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
United fans will no doubt note the rich irony in the fact that these days McTominay is basically doing the job they once paid Paul Pogba to do. Marauding
midfield engine and late-running goal threat; the all-action superhero who can chip in with 10 goals a season and pull off slide tackles that make the highlight
reels. It never quite worked out for Pogba at United, largely – if you believe his most vehement critics – on account of his hair. McTominay is a different kind
of player with a different kind of skillset. But for all the undoubted leadership qualities of Bruno Fernandes and Harry Maguire, tactically it is McTominay
running the show right now: a development with numerous interesting consequences.

Here he scored his fifth and sixth goals of the season. Previously he has never got more than seven. Where is this coming from?
</item 37>

<item 38>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
5 Ratcliffe already owns Nice in France and Lausanne in Switzerland Credit: AFP

Initially it was hoped the bidding process would be concluded in time for the summer transfer window, but the longer it dragged on, prompting several
large-scale protests from United fans, the more apparent it became that Joel and Avram were reluctant sellers, as revealed by talkSPORT in March.

Sheikh Jassim’s public statement promising to restore United ‘to former glories’ was also seen as thinly-veiled swipe at the Glazers, which did not help his
negotiating hand.

The Qatari banker’s patience finally snapped and his withdrawal from the bidding process at the end of last week has left the pathway clear for Ratcliffe to
complete what he has always viewed as his dream purchase.

WHAT ARE THE NEXT STEPS FOR RATCLIFFE?

talkSPORT understands Ratcliffe, having originally wanted a much larger up-front stake, has an agreement in principle to buy a 25 per cent share in United.
</item 38>

<item 39>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."
</item 39>

<item 40>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
He was heavily rumoured to be in line for the France position once the 2022 World Cup finished, but Didier Deschamps surprised many by signing a new contract.

Zidane would be a top choice for United, as he would be for many top clubs with a managerial vacancy, but there's questions around whether he would be open to
signing on. The 51-year-old told Marca back in June of 2022 that he isn't exactly thrilled about a Premier League job due to the language barrier.

When asked specifically about a possible job opening in Manchester, Zidane said: "I understand English, but I don't master it completely. I know that there are
coaches who go to clubs without speaking the language, but I work in a different way. Many elements come into play to win. It is a global context." Does he
still feel the same way?
</item 40>

<item 41>
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
</item 41>

<item 42>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
The match on Wednesday is not a knockout game but in truth the scenario is not much different. Defeat against Galatasaray would again end United’s Champions
League’s campaign prematurely while a win, assuming Copenhagen lose against Bayern Munich, would put qualifying for the latter stages back in their hands.
United will face the same atmosphere as they did 30 years ago because the home fans know victory would turn their subsequent trip to Denmark into a potential
shootout for who goes through with Bayern, with the achievement of knocking out a club with the storied history of United only adding to the pleasure.

In 1993 everyone was on a learning curve at United. It was their first return to Europe’s top table after an absence of 24 years. Much of the squad had won the
European Cup Winners’ Cup in 1990‑91 but this was a clear step up.
</item 42>

<item 43>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
We remember Onana also diving out of the way of the ball in Istanbul. Casemiro’s red card, lost leads in Manchester, Marcus Rashford unfairly convicted of a
venomous stamp in Copenhagen, when frankly, he’s just not that kind of semi-interested footballer.

And yet with both group games blank at half-time United were still just one goal away here from going through to the last 16, which really would have made for
an excellent punchline. It wasn’t to be. For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
Kimmich‑Goretzka‑Musiala is an A grade version of Amrabat‑McTominay‑Fernandes. Manuel Neuer is the thing Onana is trying so earnestly to be. Bayern actually did
go ahead and sign Harry Kane.

Harry Kane (centre) and his Bayern Munich teammates applaud fans after their Champions League victory at Old Trafford.
</item 43>

<item 44>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 44>

<item 45>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Varane was summoned from his inexplicable exile, after a 10-game absence when both Luke Shaw and Jonny Evans were preferred at centre-back to a World Cup
winner. He performed with an assurance to make his omission all the odder, but it has scarcely been the only U-turn in selection; with Harry Maguire and Scott
McTominay, Ten Hag is compiling a group who were first out of his plans and then back in. For a manager who likes to project an image of strong, decisiveness
leadership, he has had to reverse course rather too often. But then his team are going backwards.

When Coman scored, United became the first Premier League side to concede 15 goals in a Champions League group stage. When Maguire hobbled off, holding his
groin, and Evans came on, United had a man who played in United’s 2007-08 Champions League-winning campaign, dragooned into service 16 years later.
</item 45>

<item 46>
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
</item 46>

<item 47>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
The 42-year-old has also led the Red Devils before on a caretaker basis, installed back in 2021 after Ole Gunnar Solskjaer's dismissal and leading the club to
some big wins over Arsenal in league play and Villarreal in the Champions League before Ralf Rangnick took over.

Hansi Flick

The former German national-team manager only lasted two years on the job with Die Mannschaft, before their failures at the 2022 World Cup caught up with him.
Yet Flick has big club experience, having led Bayern Munich first as an assistant under Niko Kovac and then for two years as manager, winning the treble in
2019/20 en route to being named UEFA Coach of the Year.

Yet Flick's lack of experience of any kind outside Germany — he has just one year abroad, as an assistant at Red Bull Salzburg — coupled with his struggles at
the national-team level could give some pause at bringing him to England for such a project position.
</item 47>

<item 48>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
As and when it does get as far as a vote, Ratcliffe can be confident it will be ratified seeing as half of the 12-strong board are members of the Glazer family
and in the event of a split vote, Joel and Avram as co-chairmen will have the deciding say.

Even once the final deal is agreed, it is likely to take at least a couple of months for Ratcliffe to pass the Premier League’s owners and directors test, which
has been made more stringent since Portsmouth became the first top-flight side to go into administration in 2010.

5 Ratcliffe, one of Britain's wealthiest men, previously tried to buy Chelsea

WHAT DOES THE FUTURE LOOK LIKE UNDER RATCLIFFE’S INEOS GROUP?

Ratcliffe is demanding an executive role and overall control of day-to-day football matters and plans to appoint his own sporting director, with former
Tottenham and Southampton transfer guru Paul Mitchell one of the main names in the frame.
</item 48>

<item 49>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 49>

<item 50>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.
</item 50>

<item 51>
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
</item 51>

<item 52>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Some inside the club believe Murtough will either be moved sideways or follow Arnold through the exit door. Ratcliffe might have signalled his assessment of him
when, during a March tour of the club, he questioned the signing of a 30-year-old Casemiro on a four‑year deal worth about £350,000 a week. Murtough, whose
football department has a transfer veto, and Ten Hag, who has the same, were responsible for the Brazilian’s signing. The word is that Murtough, present when
Ratcliffe raised this, was hardly impressed. Particularly as Nice, the French club owned by Ratcliffe, recruited Aaron Ramsey, Kasper Schmeichel and Ross
Barkley as part of the nine signings in the same summer Casemiro joined United. Schmeichel was 35 and signed a three-year contract in a £1m transfer from
Leicester; Ramsey, then 31, and Barkley, then 28, came on frees and were given one-year deals.
</item 52>

<item 53>
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
</item 53>

<item 54>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
In November, it will have been a year since the deeply unpopular Glazer family left Manchester United fans rejoicing by announcing they were open to offers for
England’s most iconic football club.

Eleven months on and the saga that, at times has resembled an X-Factor style judges houses audition, may finally be coming to an end as British petrochemicals
billionaire Sir Jim Ratcliffe closes in on a partial takeover.

5 Sir Jim Ratcliffe has an offer on the table for 25 per cent stake of Manchester United Credit: AFP

Ratcliffe’s £1.3billion offer for a 25 per stake in the club he has supported since childhood will be high on the agenda when United’s board of directors meet
on Thursday, but how did we get to this point, what happens next and what could the future look like at Old Trafford?
</item 54>

<item 55>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 55>

<item 56>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 56>

<item 57>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Nostalgia underpinned Solskjaer’s management. Ten Hag cannot call upon his personal history in such a way; the Champions League final he almost reached was with
Ajax. His past does not offer the protection that means his name will always be sung at Old Trafford.

And, 24 years after Solskjaer’s stunning intervention, another United Champions League campaign finished against Bayern Munich. Not in glory, but ignominy. Like
Solskjaer, Kingsley Coman has a winner in a Champions League final; this time the winger finished off United, condemning them to a seventh home defeat of the
season. In a sense, it may have been the most respectable: United were vaguely competent on the night, but utterly incompetent over the course of a wretched
campaign.

There was no hint of a repeat of their most famous comeback against Bayern, ensuring Solskjaer’s name will forever be bracketed with Teddy Sheringham’s.
</item 57>

<item 58>
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
</item 58>

<item 59>
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
</item 59>

<item 60>
Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote (TalkSport, 2023-10-18T21:30:00+00:00)
This was a move mainly influenced by the fact Raine had managed to broker a deal with Todd Boehly to commit to invest £4.5bn to take control of Chelsea, giving
the Glazers grand ideas they could rake in as much as DOUBLE for a global sporting institution like United.

After several rounds of bidding, it quickly became clear the race for United was a straight shootout between Oldham-born Ratcliffe, who had previously offered
£4bn to buy Chelsea, and Qatari Sheikh Jassim bin Al Thani, who also professed to being a lifelong Red Devils fan and set up the ’92 Foundation’ in tribute to
the famous class of ’92 of David Beckham and company.
</item 60>

<item 61>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Indeed, Ratcliffe wants United back at "the very top of world football".

The Sporting News has laid out all the options should Man United move on from Ten Hag and look for a new beginning, in the hopes of salvaging the 2023/24
season.

MORE: How Man United produced their worst Champions League group stage in 18 years

Man United next manager: Assessing candidates to replace Erik ten Hag

If Manchester United decide to pull the plug on Erik ten Hag's managerial tenure at Old Trafford, there aren't any perfect options for his replacement.

That allows us to speculate some as to who could slot in as the next man hoping to take United back to former glories, which still evade the club since the
departure of Sir Alex Ferguson over a decade ago.

Zinedine Zidane

The former Real Madrid manager has been out of work since departing the Spanish club in 2021, clearly waiting for the right position to open up.
</item 61>

<item 62>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

They sang the name of Erik ten Hag’s predecessor. Not in a gesture of dissent, but as a reminder of a time when Manchester United were Europe’s best and an
evening without which a certain Norwegian would never have become their manager two decades later. After all, who put the ball in the Germans’ net? Ole Gunnar
Solskjaer. But that was a different millennium, a different United. Nostalgia underpinned Solskjaer’s management.
</item 62>

<item 63>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
They will always have the past at Old Trafford, but Evans may be consigned to it; perhaps, in time, Ten Hag, too.
</item 63>

<item 64>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
19 Young Boys 1-3 RB Leipzig Sept. 19 Man City 3-1 Red Star Belgrade Oct. 4 RB Leipzig 1-3 Man City Oct. 4 Red Star Belgrade 2-2 Young Boys Oct. 25 RB Leipzig
3-1 Red Star Belgrade Oct. 25 Young Boys 1-3 Man City Nov. 7 Man City 3-0 Young Boys Nov. 7 Red Star Belgrade 1-2 RB Leipzig Nov. 28 Man City 3-2 RB Leipzig
Nov. 28 Young Boys 2-0 Red Star Belgrade Dec. 13 RB Leipzig 2-1 Young Boys Dec. 13 Red Star Belgrade 2-3 Man City

Group H

Team GP W D L GD P 1. Barcelona (Q) 6 4 0 2 +6 12 2.
</item 64>

<item 65>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Jim Ratcliffe’s move to Monaco in 2020 may have been bad news for the British taxpayer, but not for Manchester United fans.

The chemical billionaire looks set to buy a 25 per cent stake in the Premier League giants after fellow bidder, Sheikh Jassim, withdrew his offer.

6 Ratcliffe could still have a big say at United Credit: getty

6 His wisest move could be to land his old neighbour Mitchell

That news hasn’t been too warmly welcomed by United supporters, who would still have to put up with the hugely unpopular Glazer family in control of the club.

It could, though, mean some sort of progress, if the Manchester born Ratcliffe follows through with his vision to revolutionise the club.

Should Ratcliffe come in, talkSPORT understands that he would take charge of footballing matters, something the Glazers have been accused of having little to no
interest in.
</item 65>

<item 66>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Christian Pulisic’s equaliser threatened to condemn them to the Europa League. Samuel Chukwueze’s winner instead meant that spot goes to AC Milan. The group of
death went down to the dying minutes, but Newcastle have perished. They had savoured a return to Europe but there will be no continental trips for them in the
spring of 2024.

And, after Kylian Mbappe’s controversial 98th-minute leveller in Paris, they can rue the loss of another lead. Once again, they may feel entitled to believe
they should have won, but for different reasons. For much of the night, they were rampant and had Milan reeling. Yet, last season’s Champions League
semi-finalists hung in the game, scored twice and hit the woodwork twice more. For Newcastle, a third consecutive defeat may come with the sense that their
injuries came at a cost.
</item 66>

<item 67>
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
</item 67>

<item 68>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 68>

<item 69>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
The performance this season was so bad that they find themselves out of Europe altogether as the bottom team in the group, although some at the club might find
solace from that. There wasn’t much else. The second place they aimed for has instead gone to Copenhagen, a club with a fraction of United’s budget. That is
where the damage was done.

While that meant there was only ever a slim hope of going through – and the Danish side’s victory over Galatasaray ensured any win would be irrelevant – the
club’s rich legacy in this competition was built on such defiance. There was absolutely none of it here.

There was nothing much at all. Where was the energy? Where was the intensity? The Stretford End eventually implored United to “attack attack attack” but the
complete absence of any such impulses eventually left some of the stadium booing the side off at the end.
</item 69>

<item 70>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Manchester United are yet to recover from their large number of long-term injuries.

Man United vs Fulham Live: Premier League LIVE – MUN vs FUL LIVE – Manchester United are travelling to Greater London for their 11th Premier League match of the
season. Fulham will host the Red Devils in the hopes of finally winning against them for the first time since 2019. It doesn’t seem like they will get a better
chance considering Man United’s horrible run of form. However, the Red Devils came away as winners in both of their last games on road. Hence, the job will not
be easy for Fulham.

Also Read

Manchester United will have to look for a way to get back to winning after 2 major defeats. They were on a 3-match winning run in all competitions when they
faced Man City in their last Premier League match. There was little hope of winning that match considering their form. The fears came true and they lost 3-0 in
the Manchester derby.
</item 70>

<item 71>
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
</item 71>

<item 72>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
As Nick Pope had denied Milan victory in San Siro with a string of saves, Stefano Pioli’s team could testify to the difference brilliant goalkeeping can make.
Now they benefited from it. Newcastle had 17 shots, eight on target, but Maignan was only beaten once. Milan had two shots on target but Dubravka conceded
twice.

The loss of the injured Pope may have proved pivotal. But perhaps seven-time winners, and a side who reached the last four last year, displayed greater knowhow.
Maybe they simply had greater strength in depth on a night when Kieran Trippier and Anthony Gordon were taken off because they seemed exhausted.

Newcastle can nevertheless depart the competition with heads held high. One win and five points seem meagre tallies for their contribution to the competition.
They have proof it can be unforgiving, reasons to return. Their first Champions League campaign for two decades will surely not be their last for another 20
years.
</item 72>

<item 73>
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
</item 73>

<item 74>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
That he can shout down the noise in the short term while building something in the long term. Right now, in the raptured afterglow of a big win, it feels like a
gamble being won. But best to come back and check in a few days, just to make sure.
</item 74>

<item 75>
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
</item 75>

<item 76>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Another infamous European night at Old Trafford. It should be a landmark low, but almost the most deflating element of this 1-0 defeat to Bayern Munich was how
grimly routine it all felt.

Manchester United, one of the grandest and richest names in this competition, have again gone out of the Champions League at the group stage. It is the third
time it has happened since Sir Alex Ferguson, meaning it’s their most common round of exit in that period.
</item 76>

<item 77>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 77>

<item 78>
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
</item 78>

<item 79>
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
</item 79>

<item 80>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Almost certain to consult Rangnick before any approach from Ratcliffe, Mitchell may be warned off of a club where long-term thinking hasn’t existed.

But, should Ratcliffe get his man, it could signal exactly the change in fortunes he and every supporter has been hoping for at Manchester United.
</item 80>

<item 81>
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
</item 81>

<item 82>
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
</item 82>

<item 83>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
And they boast a brilliant record against the Blues of late, winning six out of their last seven meetings.

They have triumphed in their last three trips to Stamford Bridge, and haven't lost in a London derby since May 2022.

As for Chelsea, they go into their match against the Gunners hopeful of a result having won their last three games in all competitions.

Both teams face injury doubts, with Arsenal to potentially be without Bukayo Saka and William Saliba, while it remains to be seen if Reece James and Armando
Broja feature for the hosts.

9 Saka missed Arsenal's 1-0 victory over City at the Emirates last time out Credit: Getty

Sheffield United desperate for points against lacklustre Manchester United

With just one point on the board, Sheffield United's season could not be going much worse given they are rock bottom.

With a fixture against Manchester United on the cards, though, they may feel their hopes could be looking up given Erik ten Hag's men disappointing season.
</item 83>

<item 84>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
"In high ball regains, we are a top of the Premier League. In the middle [of the pitch] ball regains, we are top. The pressing is very good. So that's not the
truth, what he [Carragher] is telling."

Please use Chrome browser for a more accessible video player FREE TO WATCH: Highlights from Sheffield United’s Premier League clash with Manchester United

In a fortnight where United have picked up three straight wins against Brentford, Sheffield United and FC Copenhagen, albeit via far from convincing
performances, Christian Eriksen said United still need to find their "balance and routines" this season.

Asked about those comments, Ten Hag said: "We are in the right direction. We have a way to go. I see positives but I see negatives.

"What Christian is referring to is we are not consistently in possession on the level where we can be. Like Arsenal away, the first 30 minutes in Bayern Munich
away. We should do this on a consistent basis.
</item 84>

<item 85>
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
</item 85>

<item 86>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Which, as the German champions have not lost in 40 group-stage games in this competition and United have not beaten anyone resembling a top team this season,
always felt unlikely.

This was far from their worst performance of the season, nowhere near as bad as Bournemouth or Crystal Palace or Newcastle, or the other defeat to Newcastle.
They were not as self-destructive as they had been in Denmark and Turkey and at home to Galatasaray. But they still lost.

Harry Maguire’s injury forced him off the pitch as 36-year-old Jonny Evans replaced him (EPA)

And they still went out, with four points, in fourth place. Go back a couple of months and Raphael Varane had said United could win the Champions League. On a
night when Ten Hag seemed to revise his opinion of the quadruple Champions League winner, the defender may need a rethink. Instead, United won a lone game in it
and even that required a penalty save to deny FC Copenhagen a draw.
</item 86>

<item 87>
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
</item 87>

<item 88>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.
</item 88>

<item 89>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
They defeated Honved before needing a late intervention from Cantona to avoid a deficit heading to Istanbul, where they were greeted at the airport by a horde
of Galatasaray supporters providing a wall of noise and the infamous “Welcome to Hell” banner.

“That was our first foray in the Champions League and we were expected to go through,” says the former United defender Gary Pallister, who was injured for the
game and so had to watch proceedings from the stands. “The hostility is something [to learn from] – I never experienced anything like that in the rest of my
career. You look at that and think it can’t get any worse.”

Thirty years on United head to Turkey knowing they have rarely been convincing in the Champions League in recent years. A 3-1 away win against Paris
Saint-Germain in the 2018-19 round of 16 after losing the home leg 2-0 was the last time they won a crunch match.
</item 89>

<item 90>
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
</item 90>

<item 91>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
This is the current reality for United. They obviously have many more issues than the goalkeeper – as illustrated by the goal itself – but a strange season for
Ten Hag’s primary signing only continues. United’s problems meanwhile combined. There were so many errors leading to the goal, to go with all of those that have
characterised this entire European campaign.

That can now be talked about definitively, since it has been decisively ended. It is a low but just part of a decade of disappointment in the competition that
has seen this club rise to its most emotional heights.

This game never threatened to do that. United never threatened at all. They’d been involved in so many ludicrously chaotic matches that the biggest of all ended
up being a non-event.
</item 91>

<item 92>
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
</item 92>

<item 93>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Galatasaray knew that if they beat Copenhagen at Parken they would go through but Lukas Lerager scored the only goal in a 1-0 win for the hosts

Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's
side.

Results

Date Fixture/Result Sept. 20 Galatasaray 2-2 FC Copenhagen Sept. 20 Bayern Munich 4-3 Manchester United Oct. 3 Manchester United 2-3 Galatasaray Oct. 3 FC
Copenhagen 1-2 Bayern Munich Oct. 24 Galatasaray 1-3 Bayern Munich Oct. 24 Manchester United 1-0 FC Copenhagen Nov. 8 Bayern Munich 2-1 Galatasaray Nov. 8 FC
Copenhagen 4-3 Manchester United Nov. 29 Galatasaray 3-3 Manchester United Nov.
</item 93>

<item 94>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
He will probably remain sidelined for weeks.

Other than Casemiro, Tyrell Malacia and Amad Diallo are suffering from knee issues, Lisandro Martinez is nursing a foot injury, Jadon Sancho is out due to
disciplinary issues, and Luke Shaw is yet to recover from a muscle problem. Adding to that, Victor Lindelof is suffering from a bug that could see him miss this
weekend’s match.

On the other hand, Raphael Varane who missed last day’s match due to an illness of his own, could be back in this match. Aaron Wan-Bissaka was also able to
participate in the last match coming on as a substitute. Hence, he should be able to make a comeback. Marcus Rashford, Rasmus Hojlund, and Bruno Fernandes all
could see themselves in the starting lineup.

Fulham Team News

Fulham’s injury situation has not improved so far.
</item 94>

<item 95>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 95>

<item 96>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
No family member has given a press conference or interview during their 18-year tenure. Ratcliffe has arrived on a ticket of, partly, being the local lad made
good (Failsworth, where he was born, is next to Newton Heath, United’s birthplace) so if he proves as silent as the Americans then this will be: a) poor PR; and
b) just plain poor. Football fans can be classed as the most loyal citizens of any society but even the faith of United enthusiasts has been stretched by the
cold relationship created by the Glazers’ stance regarding communication. A tap-in, then, for Ratcliffe is to remedy this by opening and maintaining a regular
line into supporters.

The Glazer family have little relationship with the Manchester United fans – something Ratcliffe would do well to avoid.
</item 96>

<item 97>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
United have done little learning in recent times, repeating the same failures in Europe’s elite competition. It would not be a disgrace to lose against a good
Galatasaray team, especially given Erik ten Hag is missing key players, but if the worst does happen, everyone will need to come out of the experience stronger.
Regardless of the result, it should be a night of education when it comes to the demands of the Champions League.

Six of United’s squad that failed in 1993 went on to win the competition six years later. “It did galvanise us,” says Mike Phelan, who started in central
midfield 30 years ago. “Winning the Cup Winners’ Cup took us to another level and we felt pretty comfortable with each other and felt we could develop into a
serious team, and I think going into the Champions League was one of those moments where we expected to do well – the experiences along the way were great and
took us to the next level further down the line.
</item 97>

<item 98>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Manchester United managers since Sir Alex Ferguson

If Manchester United were to appoint a new manager, that person would be the ninth coach to take charge of a United team following Sir Alex Ferguson's
retirement in May 2013, including caretaker/interim managers Ryan Giggs, Ralf Rangnick and Carrick.

Manager Honours Games David Moyes Community Shield 51 Ryan Giggs (caretaker) — 4 Louis van Gaal FA Cup 103 Jose Mourinho Community Shield, Europa League, League
Cup 144 Ole Gunnar Solskjaer — 168 Michael Carrick (caretaker) — 3 Ralf Rangnick (interim) — 29 Erik ten Hag League Cup *

*Still manager of the club
</item 98>

<item 99>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.
</item 99>

<item 100>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
But it is over, and they are left to wonder about what might have been.
</item 100>
=== END ITEM id=a6fd427dba028e74 ===

=== ITEM id=ab1f7a36c62557e5 ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: Google touted Gemini's superior architecture and capabilities in blog posts and press materials.
  Fact 0.1: Google claimed that Gemini meets or exceeds the performance of other leading gen AI models.
  Fact 0.2: OpenAI's GPT-4 is one of the other leading gen AI models Google named.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 1>

<item 2>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 2>

<item 3>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 3>

<item 4>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 4>

<item 5>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 5>

<item 6>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 7>

<item 8>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 10>
=== END ITEM id=ab1f7a36c62557e5 ===
